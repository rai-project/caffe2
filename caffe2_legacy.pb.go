// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: caffe2_legacy.proto

package caffe2

import proto "github.com/gogo/protobuf/proto"
import fmt "fmt"
import math "math"

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

// This is a compile-time assertion to ensure that this generated file
// is compatible with the proto package it is being compiled against.
// A compilation error at this line likely means your copy of the
// proto package needs to be updated.
const _ = proto.GoGoProtoPackageIsVersion2 // please upgrade the proto package

type LegacyPadding int32

const (
	LegacyPadding_NOTSET LegacyPadding = 0
	// VALID and SAME are two strategies adopted in Google DistBelief: it forces
	// the input shape as follows. For SAME, the output is:
	//   R_out = ceil(float(R) / float(S))
	//   C_out = ceil(float(C) / float(S))
	// where R and C are row and column, S is the stride, and K is the kernel.
	// The number of padded pixels is then computed as
	//   Pr = ((R_out - 1) * S + K - R)
	//   Pc = ((C_out - 1) * S + K - C)
	// When Pr and Pc are even numbers, both sides (left and right, or top and
	// bottom) get half each. When Pr and Pc are odd numbers, the right and the
	// bottom gets the one additional padding pixel.
	// For VALID, padding values of 0 are always used.
	LegacyPadding_VALID LegacyPadding = 1
	LegacyPadding_SAME  LegacyPadding = 2
	// CAFFE_LEGACY_POOLING is a flag that notifies the code to use the old Caffe
	// padding strategy.
	// Basically, in caffe2, after padding the convolution and pooling use the
	// same computation strategy: half-windows at the right and bottom are
	// discarded. In Caffe, convolution follows this strategy but if there are
	// some pixels in the half-windows, the pooling layer will actually put one
	// additional output. If you set LegacyPadding to this, we will compute the
	// equivalent padding strategy in caffe2 so that the output size is
	// backward compatible with Caffe.
	// THIS IS NOW DEPRECATED. ANY non-conventional use has to be manually
	// converted.
	LegacyPadding_CAFFE_LEGACY_POOLING LegacyPadding = 3
)

var LegacyPadding_name = map[int32]string{
	0: "NOTSET",
	1: "VALID",
	2: "SAME",
	3: "CAFFE_LEGACY_POOLING",
}
var LegacyPadding_value = map[string]int32{
	"NOTSET":               0,
	"VALID":                1,
	"SAME":                 2,
	"CAFFE_LEGACY_POOLING": 3,
}

func (x LegacyPadding) Enum() *LegacyPadding {
	p := new(LegacyPadding)
	*p = x
	return p
}
func (x LegacyPadding) String() string {
	return proto.EnumName(LegacyPadding_name, int32(x))
}
func (x *LegacyPadding) UnmarshalJSON(data []byte) error {
	value, err := proto.UnmarshalJSONEnum(LegacyPadding_value, data, "LegacyPadding")
	if err != nil {
		return err
	}
	*x = LegacyPadding(value)
	return nil
}
func (LegacyPadding) EnumDescriptor() ([]byte, []int) {
	return fileDescriptor_caffe2_legacy_7a491fa696f09c9f, []int{0}
}

func init() {
	proto.RegisterEnum("caffe2.LegacyPadding", LegacyPadding_name, LegacyPadding_value)
}

func init() { proto.RegisterFile("caffe2_legacy.proto", fileDescriptor_caffe2_legacy_7a491fa696f09c9f) }

var fileDescriptor_caffe2_legacy_7a491fa696f09c9f = []byte{
	// 149 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xe2, 0x12, 0x4e, 0x4e, 0x4c, 0x4b,
	0x4b, 0x35, 0x8a, 0xcf, 0x49, 0x4d, 0x4f, 0x4c, 0xae, 0xd4, 0x2b, 0x28, 0xca, 0x2f, 0xc9, 0x17,
	0x62, 0x83, 0x08, 0x6a, 0x79, 0x71, 0xf1, 0xfa, 0x80, 0xc5, 0x03, 0x12, 0x53, 0x52, 0x32, 0xf3,
	0xd2, 0x85, 0xb8, 0xb8, 0xd8, 0xfc, 0xfc, 0x43, 0x82, 0x5d, 0x43, 0x04, 0x18, 0x84, 0x38, 0xb9,
	0x58, 0xc3, 0x1c, 0x7d, 0x3c, 0x5d, 0x04, 0x18, 0x85, 0x38, 0xb8, 0x58, 0x82, 0x1d, 0x7d, 0x5d,
	0x05, 0x98, 0x84, 0x24, 0xb8, 0x44, 0x9c, 0x1d, 0xdd, 0xdc, 0x5c, 0xe3, 0x7d, 0x5c, 0xdd, 0x1d,
	0x9d, 0x23, 0xe3, 0x03, 0xfc, 0xfd, 0x7d, 0x3c, 0xfd, 0xdc, 0x05, 0x98, 0x9d, 0x24, 0x4e, 0x3c,
	0x92, 0x63, 0xbc, 0xf0, 0x48, 0x8e, 0xf1, 0xc1, 0x23, 0x39, 0xc6, 0x09, 0x8f, 0xe5, 0x18, 0x2e,
	0x3c, 0x96, 0x63, 0xb8, 0xf1, 0x58, 0x8e, 0x01, 0x10, 0x00, 0x00, 0xff, 0xff, 0xf0, 0xb1, 0x5b,
	0xf8, 0x83, 0x00, 0x00, 0x00,
}
