// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/plugin/common_utils.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"

#include <algorithm>
#include <cstring>
#include <random>

using namespace ov::intel_gpu;

namespace {

ov::ITensor& impl(ov::Tensor& tensor) {
    return *ov::get_tensor_impl(tensor)._ptr;
}

void fill_pattern(ov::Tensor& tensor) {
    auto* bytes = static_cast<uint8_t*>(tensor.data());
    for (size_t i = 0; i < tensor.get_byte_size(); ++i) {
        bytes[i] = static_cast<uint8_t>(i * 131 + 7);
    }
}

// Element-wise reference: walk the shape and copy one element at a time through both tensors' strides
void copy_elementwise(const ov::Tensor& src, ov::Tensor& dst) {
    const auto& shape = src.get_shape();
    const auto& src_strides = src.get_strides();
    const auto& dst_strides = dst.get_strides();
    const auto element_size = src.get_element_type().size();
    const auto* src_data = static_cast<const uint8_t*>(src.data());
    auto* dst_data = static_cast<uint8_t*>(dst.data());
    ov::Shape index(shape.size(), 0);
    for (size_t e = 0; e < src.get_size(); ++e) {
        size_t src_offset = 0;
        size_t dst_offset = 0;
        for (size_t d = 0; d < shape.size(); ++d) {
            src_offset += index[d] * src_strides[d];
            dst_offset += index[d] * dst_strides[d];
        }
        std::memcpy(dst_data + dst_offset, src_data + src_offset, element_size);
        for (size_t d = shape.size(); d-- > 0;) {
            if (++index[d] < shape[d]) {
                break;
            }
            index[d] = 0;
        }
    }
}

}  // namespace

TEST(copy_strided_test, roi_view_to_packed) {
    const ov::Shape parent_shape{1, 3, 16, 16};
    const ov::Coordinate begin{0, 0, 4, 6};
    const ov::Coordinate end{1, 3, 12, 14};
    ov::Tensor parent(ov::element::u8, parent_shape);
    fill_pattern(parent);
    auto roi = ov::Tensor(parent, begin, end);
    ASSERT_FALSE(roi.is_continuous());

    ov::Tensor packed(ov::element::u8, roi.get_shape());
    copy_strided(impl(roi), impl(packed));

    const auto* parent_data = static_cast<const uint8_t*>(parent.data());
    const auto* packed_data = static_cast<const uint8_t*>(packed.data());
    const auto strides = parent.get_strides();  // byte strides equal element strides for u8
    size_t idx = 0;
    for (size_t n = begin[0]; n < end[0]; ++n) {
        for (size_t c = begin[1]; c < end[1]; ++c) {
            for (size_t y = begin[2]; y < end[2]; ++y) {
                for (size_t x = begin[3]; x < end[3]; ++x, ++idx) {
                    ASSERT_EQ(packed_data[idx], parent_data[n * strides[0] + c * strides[1] + y * strides[2] + x * strides[3]])
                        << "at [" << n << "," << c << "," << y << "," << x << "]";
                }
            }
        }
    }
    ASSERT_EQ(idx, packed.get_size());
}

TEST(copy_strided_test, packed_to_roi_view_leaves_rest_of_parent_untouched) {
    const ov::Shape parent_shape{1, 3, 16, 16};
    const ov::Coordinate begin{0, 0, 4, 6};
    const ov::Coordinate end{1, 3, 12, 14};
    const float untouched = -1.f;
    ov::Tensor parent(ov::element::f32, parent_shape);
    std::fill_n(parent.data<float>(), parent.get_size(), untouched);
    auto roi = ov::Tensor(parent, begin, end);

    ov::Tensor packed(ov::element::f32, roi.get_shape());
    auto* packed_data = packed.data<float>();
    for (size_t i = 0; i < packed.get_size(); ++i) {
        packed_data[i] = static_cast<float>(i);
    }
    copy_strided(impl(packed), impl(roi));

    const auto* parent_data = parent.data<float>();
    auto strides = parent.get_strides();
    for (auto& stride : strides) {
        stride /= sizeof(float);
    }
    size_t idx = 0;
    for (size_t n = 0; n < parent_shape[0]; ++n) {
        for (size_t c = 0; c < parent_shape[1]; ++c) {
            for (size_t y = 0; y < parent_shape[2]; ++y) {
                for (size_t x = 0; x < parent_shape[3]; ++x) {
                    const bool inside = n >= begin[0] && n < end[0] && c >= begin[1] && c < end[1] &&
                                        y >= begin[2] && y < end[2] && x >= begin[3] && x < end[3];
                    const float expected = inside ? static_cast<float>(idx++) : untouched;
                    ASSERT_EQ(parent_data[n * strides[0] + c * strides[1] + y * strides[2] + x * strides[3]], expected)
                        << "at [" << n << "," << c << "," << y << "," << x << "]";
                }
            }
        }
    }
    ASSERT_EQ(idx, packed.get_size());
}

TEST(copy_strided_test, contiguous_tensors) {
    ov::Tensor src(ov::element::i64, ov::Shape{2, 3, 5});
    fill_pattern(src);
    ov::Tensor dst(ov::element::i64, src.get_shape());
    copy_strided(impl(src), impl(dst));
    ASSERT_EQ(0, std::memcmp(src.data(), dst.data(), src.get_byte_size()));
}

TEST(copy_strided_test, random_views_match_elementwise_reference) {
    std::mt19937_64 rng(42);
    const std::vector<ov::element::Type> types{ov::element::u8, ov::element::f16, ov::element::f32, ov::element::i64};
    for (int iteration = 0; iteration < 2000; ++iteration) {
        const size_t rank = 1 + rng() % 5;
        const auto& type = types[rng() % types.size()];

        // Source: an ROI view of a random parent
        ov::Shape src_parent_shape(rank), roi_shape(rank);
        ov::Coordinate src_begin(rank), src_end(rank);
        for (size_t d = 0; d < rank; ++d) {
            src_parent_shape[d] = 1 + rng() % 6;
            src_begin[d] = rng() % src_parent_shape[d];
            src_end[d] = src_begin[d] + 1 + rng() % (src_parent_shape[d] - src_begin[d]);
            roi_shape[d] = src_end[d] - src_begin[d];
        }
        // Destination: packed, or an ROI view of another parent that is at least as large as the ROI
        const bool dst_packed = rng() % 3 == 0;
        ov::Shape dst_parent_shape(rank);
        ov::Coordinate dst_begin(rank), dst_end(rank);
        for (size_t d = 0; d < rank; ++d) {
            dst_parent_shape[d] = dst_packed ? roi_shape[d] : roi_shape[d] + rng() % 4;
            dst_begin[d] = dst_packed ? 0 : rng() % (dst_parent_shape[d] - roi_shape[d] + 1);
            dst_end[d] = dst_begin[d] + roi_shape[d];
        }

        ov::Tensor src_parent(type, src_parent_shape);
        fill_pattern(src_parent);
        auto src = ov::Tensor(src_parent, src_begin, src_end);

        ov::Tensor dst_parent(type, dst_parent_shape);
        ov::Tensor ref_parent(type, dst_parent_shape);
        std::memset(dst_parent.data(), 0xEE, dst_parent.get_byte_size());
        std::memset(ref_parent.data(), 0xEE, ref_parent.get_byte_size());
        auto dst = ov::Tensor(dst_parent, dst_begin, dst_end);
        auto ref = ov::Tensor(ref_parent, dst_begin, dst_end);

        copy_strided(impl(src), impl(dst));
        copy_elementwise(src, ref);

        ASSERT_EQ(0, std::memcmp(dst_parent.data(), ref_parent.data(), dst_parent.get_byte_size()))
            << "iteration " << iteration << " rank " << rank << " type " << type << " roi " << roi_shape;
    }
}
