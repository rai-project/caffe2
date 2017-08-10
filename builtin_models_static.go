// Code generated by go-bindata.
// sources:
// builtin_models/Squeezenet.yml
// builtin_models/bvlc_googlenet.yml
// DO NOT EDIT!

package caffe2

import (
	"bytes"
	"compress/gzip"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"time"
)

func bindataRead(data, name string) ([]byte, error) {
	gz, err := gzip.NewReader(strings.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("Read %q: %v", name, err)
	}

	var buf bytes.Buffer
	_, err = io.Copy(&buf, gz)
	clErr := gz.Close()

	if err != nil {
		return nil, fmt.Errorf("Read %q: %v", name, err)
	}
	if clErr != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

type asset struct {
	bytes []byte
	info  os.FileInfo
}

type bindataFileInfo struct {
	name    string
	size    int64
	mode    os.FileMode
	modTime time.Time
}

func (fi bindataFileInfo) Name() string {
	return fi.name
}
func (fi bindataFileInfo) Size() int64 {
	return fi.size
}
func (fi bindataFileInfo) Mode() os.FileMode {
	return fi.mode
}
func (fi bindataFileInfo) ModTime() time.Time {
	return fi.modTime
}
func (fi bindataFileInfo) IsDir() bool {
	return false
}
func (fi bindataFileInfo) Sys() interface{} {
	return nil
}

var _squeezenetYml = "\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff\x84\x55\x4b\x8f\xdb\x36\x10\xbe\xeb\x57\x0c\x60\x04\x68\x01\xaf\x64\xef\x1b\x3a\x14\x48\xf7\x94\x8b\x0f\xe9\xb1\x08\x8c\x31\x35\xb2\xd8\xf0\x15\x72\xb8\xb6\xf3\xeb\x8b\xa1\x64\xcb\x8b\x6c\xbb\x3e\x18\x22\xe7\x9b\xd7\xc7\x8f\x43\x87\x96\x5a\xf8\xeb\x47\x26\xfa\x49\x1b\x62\x58\x80\x6c\x81\xef\xe1\xe4\x73\x04\xeb\x3b\x32\x55\x1f\xd1\xd2\xc1\xc7\xef\x6d\x05\x30\xba\xbc\x60\xdf\xd3\x2d\x2c\xe0\x62\x83\xde\x47\xe0\x81\x26\x1f\x80\x57\x8a\x49\x7b\xd7\xc2\xaa\x7e\xae\xd7\x6f\xa0\x93\x09\x94\x77\x1c\x51\x3b\xae\x2e\xe0\x75\xbd\x82\xc5\x05\xa0\x5d\xef\xa3\x45\x1e\xbf\x21\x91\x45\xc7\x5a\x5d\xec\xa3\xb5\x92\x38\xa8\x1d\xc5\x16\x16\x70\x59\x24\xc8\x89\x3a\x60\x0f\x81\xa2\x20\xc7\xd2\x20\x44\xea\xb4\x92\x98\x15\xcc\xbf\x05\xd8\x6c\x58\x07\x43\x10\x0c\xb2\xe0\x13\x28\x74\xb0\x23\x48\x81\x94\xee\x35\x75\x15\x00\xda\xee\xf1\xbe\x2d\x9e\xfb\x90\x5b\x88\xa8\x43\xf4\xff\x90\xe2\x46\x61\xb4\xe6\x46\x15\x6a\xda\x82\xbb\x51\x21\x17\xa8\xfa\x18\xba\x2f\xd0\x10\xd4\xe3\xbd\xa1\xf6\x63\xaf\x09\x39\xf9\x7d\x50\xcd\x35\xb8\xa3\xa4\xa2\x0e\x5c\xf8\xfe\xa3\x82\xf3\xf9\x3b\x62\x40\x35\x68\x7a\xa5\x04\x49\x5b\x6d\x30\x42\xa4\x94\x0d\x27\x61\xf1\xb3\xa1\xe3\x86\x78\x09\xc8\xf0\xb0\x3a\x42\x4f\x07\x8a\x10\x50\x8e\x95\x85\x6e\x74\x1d\xac\x9b\x87\xd5\x8a\x87\xa2\x84\xa4\x7f\x52\x2d\xf1\x2d\x1a\x33\x92\x9f\x00\xa3\x48\x24\x12\xf4\x84\x49\xef\x0c\x49\xe8\x8e\x82\xf1\x27\xf0\x0e\x06\x8c\xdd\x41\x30\x07\xcd\x03\x18\x6d\x35\x53\x07\x96\xac\x8f\xa7\x25\x44\xfa\x91\x75\x24\x30\x94\x52\x05\xa0\xbc\xb5\xd9\x69\x35\x0a\xa4\xcb\x51\xbb\x3d\x74\x3a\x71\xd4\xbb\x2c\x7e\x45\x5d\xda\xed\x97\xa5\x36\x09\x2b\x49\x29\x96\x9c\x17\x9c\xac\x94\xd1\xe4\x38\xd5\x33\x1d\x9b\x6b\x3a\x1e\x9e\xea\x87\x4f\x25\xc8\xf3\xaa\xbe\xfb\x04\xec\xc3\xcd\xba\xac\xe5\xeb\x01\x50\xe5\x88\x8a\x4b\x0b\x5f\x2c\xee\xc5\xbb\xae\x22\xf5\x14\xc9\x29\x4a\x22\xcc\x79\x55\x34\x89\x41\x38\x6b\xe0\x40\xbb\xa4\x99\xe4\x93\x58\xd5\x35\x8c\xe7\xb3\x93\x5e\xae\xef\xd3\x0d\x0c\xcc\x21\xb5\x4d\x83\xf1\xa8\x5f\x6b\x1f\xf7\x4d\xe8\xfa\x66\xfd\xb8\xba\xad\x57\x4f\x77\x8f\xab\xd7\xfb\x3a\x74\xfd\x1b\xe8\x5e\xf3\x90\x77\xb5\xf2\xb6\x19\x95\xd0\x8c\xc7\xd0\x70\x24\x6a\x2c\x26\xa6\xd8\xa4\xcb\xf9\x57\x0b\x30\x5a\x91\x4b\x65\x04\xcc\xd9\xa7\xcd\x16\xfe\xfc\xfc\xe5\x6b\xb5\x00\xed\x42\x1e\x45\x31\x63\xc6\x3d\x11\xee\x02\x7a\x1d\x13\x8f\x28\xe0\x53\xa0\x5f\x86\xc3\x4d\xd9\x6e\x41\x0b\x57\xd5\x78\xff\xae\x74\x79\xce\x7e\x15\xa7\x80\xde\x48\x57\x00\x63\x8a\x39\xca\xac\x46\x61\xbc\xa4\x9e\xb7\xa6\xeb\xde\x69\x4b\x4e\xc6\x47\x6a\xe1\xef\xf5\x12\xee\x96\x70\x7b\xfb\x54\xfe\xbe\x4d\x10\x4b\xe8\xc4\x78\xfb\xbc\x84\xf3\xdf\xb7\xca\x67\x0e\x99\xc7\x0e\x25\x79\x09\x3f\x55\x3a\xda\x2a\x98\xfa\xea\x09\x39\x47\x2a\x50\x7c\xaf\xb3\x11\x3f\x17\x57\xbd\xd3\xdc\x84\x31\xb8\x2b\x9c\x5d\xf5\x36\x31\xf6\x5e\x7f\x53\xe6\xb4\xcd\xd1\xb4\x45\x07\x6d\xd3\x74\xc8\x58\x77\xd6\xa8\xda\x9a\xc6\x1e\x1d\xf1\x59\x07\x85\x3a\x59\xa7\x93\x4b\xc4\x35\x1f\xb9\x2a\x26\xe1\xef\x3c\xf7\xd2\x34\x39\xf7\x11\xc3\x50\x54\x7f\x20\xbd\x1f\x38\xc9\x78\xf0\x39\x2a\x92\xdc\x3b\x4c\x34\x67\xfd\x5f\xf5\x45\x3c\xbc\x23\x3e\x18\x13\x6c\x03\xf2\xd0\x9e\xc7\xf4\xd6\x11\xd7\x61\x57\xc1\x39\xe7\x64\xa6\x23\xa9\xd9\xa6\xd3\x16\xa3\x1a\xf4\xab\x70\x8f\x26\x11\x2c\x40\xf7\x90\x64\x5c\xf1\x40\xae\xf0\x79\x2e\x10\x74\x02\x04\xf9\x60\x0f\xe8\x60\xf2\xbc\x7e\x0d\xae\x9e\x05\xf1\x9c\xeb\xba\xee\x7e\xdc\x28\xe1\x3b\x72\x5e\x06\xc9\xf0\x5f\x51\x7a\x6d\xa8\xbc\x9d\xe9\xac\x80\x5f\xc9\x94\x91\xa7\xc7\x52\xcf\x25\x21\x4f\x33\xaa\xe8\x99\x8e\x1c\x11\x1c\x71\x79\x45\x67\x5b\x05\xf0\x5d\xbb\xae\x85\x97\xcd\x66\xaa\x58\xd6\x92\xc9\x51\x8e\x68\x2e\x3e\xbf\xbd\x6c\x36\x4b\xf8\x2a\x7f\x75\x5d\xff\x2e\x7a\x9d\x46\xe4\x56\x34\x92\x88\xdb\xcb\xfc\x92\x2b\x39\xee\x5d\x1e\xd2\x72\x8d\x27\x87\x0a\xc0\xa2\xd3\x3d\x25\xde\x62\xe6\xc1\xc7\x16\x70\xd7\x65\xd3\x55\xff\x06\x00\x00\xff\xff\xd1\xf2\xa0\x13\x57\x08\x00\x00"

func squeezenetYmlBytes() ([]byte, error) {
	return bindataRead(
		_squeezenetYml,
		"Squeezenet.yml",
	)
}

func squeezenetYml() (*asset, error) {
	bytes, err := squeezenetYmlBytes()
	if err != nil {
		return nil, err
	}

	info := bindataFileInfo{name: "Squeezenet.yml", size: 2135, mode: os.FileMode(420), modTime: time.Unix(1502399948, 0)}
	a := &asset{bytes: bytes, info: info}
	return a, nil
}

var _bvlc_googlenetYml = "\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff\xb4\x56\x4d\x6f\xdb\x38\x13\xbe\xeb\x57\x0c\x62\x14\x48\xde\xd7\x96\x6c\xd7\x4d\x53\x15\xd8\xc3\x66\x81\xc5\x02\x45\x0e\xfb\x79\x58\x14\xc6\x88\x1a\x49\x6c\x28\x92\x4b\x0e\xed\xb8\xbf\x7e\x41\x4a\xb2\x9c\x36\x45\x4f\xeb\x83\x21\x72\x9e\xf9\xe4\xc3\x19\x6a\xec\xa9\x84\x1f\xff\xfc\x70\xbf\xfa\xd9\x98\xf6\x03\x3d\x10\xc3\x02\xe2\x36\x98\x06\x4e\x26\x38\xe8\x4d\x4d\x2a\x6b\x1c\xf6\x74\x34\xee\xb1\xcc\x00\x06\xb5\x7b\x6c\x1a\xda\xc2\x02\xce\x32\x68\x8c\x03\xee\x68\xd4\x01\x38\x90\xf3\xd2\xe8\x12\xd6\xf9\x5d\xbe\x79\x06\x1d\x45\x20\x8c\x66\x87\x52\x73\x76\x06\x6f\xf2\x35\x2c\xce\x00\xa9\x1b\xe3\x7a\xe4\xe1\x1b\x3c\xf5\xa8\x59\x8a\xb3\x7c\x90\x66\xd1\x0e\x4a\x4d\xae\x84\x05\x9c\x17\x1e\x82\xa7\x1a\xd8\x80\x25\x17\x91\x43\x68\x60\x1d\xd5\x52\x44\x9b\x19\xcc\xbf\x05\xf4\x41\xb1\xb4\x8a\xc0\x2a\xe4\x88\xf7\x20\x50\x43\x45\xe0\x2d\x09\xd9\x48\xaa\x33\x00\xec\xeb\xdb\x5d\x99\x34\x5b\x1b\x4a\x70\x28\xad\x33\x9f\x48\x70\x21\xd0\xf5\x6a\x25\x52\x69\xca\x84\x5b\x09\x1b\x12\x54\x7c\x1f\xda\x26\xa8\xb5\xe2\x76\xa7\xa8\xfc\xbe\xd6\x88\x1c\xf5\xbe\x13\xcd\x25\xb8\x26\x2f\x9c\xb4\x9c\xea\xfd\x43\x06\xf0\x7b\x27\xfd\x58\x1b\xe9\x01\xc1\x91\x55\x52\x0c\x55\x37\xcd\x7c\xa8\x30\x68\x56\x54\xc7\xc3\x88\xdb\x91\x38\x2a\x11\xc7\x86\x6a\xd2\xc9\xe1\x2f\x82\xa3\x09\xaa\x06\x25\x1f\x29\x1e\x00\x77\xa8\x1f\xe1\xbe\x73\xd2\xb3\x44\x0d\xbf\x7d\xa6\x96\xea\x53\xe2\x0c\x2a\x05\x31\x80\x8e\x94\x9d\xec\x7e\x11\xc1\xec\x26\xc5\x91\x67\x00\x3f\xc9\xa6\x21\x47\x5a\x90\x4f\xac\x34\x0c\x89\x4a\x52\xb7\x70\x94\xdc\x8d\x66\x94\x6c\x3b\x8e\x7b\x35\x32\xae\x30\xb4\x3d\x69\x4e\x76\xdf\x7f\x53\xcb\x0b\x54\x04\x31\xb2\x78\xee\xbc\x72\x11\xff\xb2\x81\xe0\xc9\xc3\xd5\x13\x1e\x24\xb9\xab\x98\xa8\xd4\x92\x25\x2a\xf9\x99\x92\xa9\x23\x45\xff\x1e\xa4\xf6\x4c\x58\xc7\x5c\xae\x5a\x0c\xde\x4b\xd4\x57\xd1\xc0\x3f\x41\x8a\xc7\xbd\x37\xea\x40\x2e\xb7\xce\xb0\xe1\x27\x1e\xcc\x22\xd4\x63\x8e\x0c\x8a\xd0\xa5\x20\x1d\x32\x41\x4d\x02\x4f\x60\x8d\x92\xe2\x94\x4a\x9b\x7c\x19\x27\x5b\xa9\x51\xc1\x17\xd6\x96\x11\xc2\xb1\xcc\xe6\x18\xad\xf6\x41\x74\xd0\xa0\x67\x72\x73\xf2\xd7\xb7\x6b\x20\x6b\x44\xe7\xe1\xe0\x61\xfb\x66\x5a\xdd\xbc\x4f\xfc\x20\xa8\x82\xae\x15\xd5\x33\x4d\xa2\x4b\xc9\xe4\x86\x43\xda\x2e\x77\xeb\xf5\x72\xbd\x5e\x83\xd7\x68\x7d\x67\xf8\xc2\xe4\x0d\x04\x1f\x9d\xbc\x98\xec\xc4\xbf\xe7\x0e\x4c\x15\xef\x70\x0c\x97\x8d\x5d\x6d\x00\x85\x08\x0e\xc5\x09\x6e\xef\xf2\xb7\xaf\xe0\xfa\xf5\x26\x7f\xfd\x0a\xc8\x39\xe3\x6e\x00\x75\x3d\x02\xdf\xcc\xc0\xbb\xbb\xfc\xdd\x2b\xb8\xde\x6c\xf2\xcd\x19\x68\x86\x4a\x1d\x50\xc9\x7a\x88\xdb\x13\x2f\xc7\xe8\x3e\x05\xcf\x49\x2c\x48\xc7\xda\x08\x67\x6c\x0e\xd7\x7f\x24\x61\xdc\xc7\x03\x39\x6c\x53\x73\xdc\xac\x93\xd8\x2f\xe1\x7a\x07\xff\x87\xcd\xa8\x73\x03\xff\x83\x2d\xf4\x32\x7a\x5b\x82\xef\xd2\x1d\x18\x52\x01\x84\x4a\x32\x74\xb2\xed\xc8\x9d\xa3\xcc\x6f\x32\x47\x67\x1e\xc3\x02\xe6\x55\xea\x5b\x68\x63\x1b\x2b\xe0\x48\x95\x97\x4c\xf1\x93\x58\xe4\xf9\x74\x13\xa7\xd0\xa6\x9e\xbb\x82\x8e\xd9\xfa\xb2\x28\xd0\x3d\xc9\x43\x6e\x5c\x5b\xd8\xba\x29\x36\xbb\xf5\xbb\x7c\x77\xb7\xdb\xe6\xb6\x6e\x9e\xe1\x5a\xc9\x5d\xa8\x72\x61\xfa\x22\x8e\x82\x22\xf5\x8b\x82\x1d\x51\xd1\x27\x8e\x14\xc9\xb6\x2f\xaa\x83\x12\xfb\x36\x5d\x44\x4d\x9c\x2d\x40\x49\x41\xda\xd3\xb3\x0e\x91\x8d\x9b\x25\x04\xed\xc8\xb3\x93\x82\xa9\xce\x16\x20\xb5\x0d\xec\x87\x56\x30\x61\x87\xbd\x78\x7b\x17\xd0\x48\xe7\x79\x40\x01\x9f\x2c\x7d\x35\x4d\x56\x69\xbb\x04\xd9\x63\x4b\xd9\xd0\xb0\x2f\x1a\xd9\x14\xc5\x85\x9d\x04\x7a\xd6\xeb\x12\x65\x93\x8b\xd9\x8a\xc5\x38\x95\x98\x5c\x2a\x7f\x72\x3d\x6f\x8d\xf3\xa1\x96\x3d\xe9\x38\x6f\x7c\x09\x7f\x6f\x96\xf0\x7a\x09\xdb\xed\xdb\xf4\xf7\x71\x84\xf4\x84\x3a\x0a\xb7\x77\x4b\x98\xfe\x3e\x66\x26\xb0\x0d\x3c\x64\x18\x9d\x27\xf3\x63\xa4\x83\x2c\x83\x31\xaf\x86\x90\x83\xa3\x04\xc5\x97\x32\x1b\xf0\x73\x70\xd9\x0b\xc9\x8d\x18\x85\x55\xaa\xd9\x45\x6e\x63\xc5\x5e\xca\x6f\xf4\xec\xf7\xc1\xa9\x32\xf1\xa2\x2c\x8a\xd8\xed\xf2\xba\x57\x22\xef\x55\xd1\x3f\x69\xe2\x89\x07\xa9\x74\x71\xed\x4f\xda\x13\xe7\xf1\x0a\x27\x51\xac\xdf\x34\x28\xa7\x71\xd2\x3a\xb4\x5d\xba\x9f\x53\x27\x74\xe4\x4d\x70\x82\xa2\xef\x24\xdd\x5b\xe4\xae\x7c\x89\x8f\x0e\xe5\xea\xf9\x30\x1b\x43\x70\x78\x9c\xc8\x39\x8c\xb7\x2f\xc8\x59\x8c\x13\x7e\xaf\x89\x73\x5b\x65\x30\x79\xff\x2f\x7c\xc5\x8e\x3f\x3b\x92\x7e\x8f\x4e\x74\xf2\x10\x8f\x14\x95\x27\x58\x80\x6c\x86\x36\xc3\x1d\x0d\xfd\xa7\x42\x4f\xb1\xda\xc3\xb4\x8d\x1f\x6c\x00\x35\x8c\x9a\x97\xaf\x92\x8b\xe7\x49\xd4\x9c\x2b\x76\x59\xd4\x61\x23\x99\xaf\x49\x1b\x4e\xb3\xe7\x1b\x56\x1a\xa9\x28\xbd\xe1\xfc\x44\xac\xaf\xcf\x28\x4e\xc2\x71\x10\x4f\x21\x21\xb3\x93\x55\xe0\xa1\x4b\xd1\x13\x3b\x04\x4d\x9c\x5e\x73\xb3\x2c\x03\x78\x94\xba\x2e\xe1\xfe\xe1\x61\x8c\x38\xae\xa3\x27\x4d\xc1\xa1\x3a\xeb\x5c\xdf\x3f\x3c\x2c\xe1\xd7\xf8\x97\xe7\xf9\x4d\xbc\x06\xe3\x28\xda\x47\xea\x79\xe2\x12\x7e\x89\x4c\x1b\x1e\xa5\xe3\xde\xf9\x41\x97\xba\xc3\xa8\x90\x01\xf4\xa8\x65\x43\x9e\xf7\x18\xb8\x33\xae\x04\xac\xea\xa0\xea\xec\xdf\x00\x00\x00\xff\xff\x84\x4f\xa0\x4c\xe3\x0a\x00\x00"

func bvlc_googlenetYmlBytes() ([]byte, error) {
	return bindataRead(
		_bvlc_googlenetYml,
		"bvlc_googlenet.yml",
	)
}

func bvlc_googlenetYml() (*asset, error) {
	bytes, err := bvlc_googlenetYmlBytes()
	if err != nil {
		return nil, err
	}

	info := bindataFileInfo{name: "bvlc_googlenet.yml", size: 2787, mode: os.FileMode(420), modTime: time.Unix(1502402249, 0)}
	a := &asset{bytes: bytes, info: info}
	return a, nil
}

// Asset loads and returns the asset for the given name.
// It returns an error if the asset could not be found or
// could not be loaded.
func Asset(name string) ([]byte, error) {
	cannonicalName := strings.Replace(name, "\\", "/", -1)
	if f, ok := _bindata[cannonicalName]; ok {
		a, err := f()
		if err != nil {
			return nil, fmt.Errorf("Asset %s can't read by error: %v", name, err)
		}
		return a.bytes, nil
	}
	return nil, fmt.Errorf("Asset %s not found", name)
}

// MustAsset is like Asset but panics when Asset would return an error.
// It simplifies safe initialization of global variables.
func MustAsset(name string) []byte {
	a, err := Asset(name)
	if err != nil {
		panic("asset: Asset(" + name + "): " + err.Error())
	}

	return a
}

// AssetInfo loads and returns the asset info for the given name.
// It returns an error if the asset could not be found or
// could not be loaded.
func AssetInfo(name string) (os.FileInfo, error) {
	cannonicalName := strings.Replace(name, "\\", "/", -1)
	if f, ok := _bindata[cannonicalName]; ok {
		a, err := f()
		if err != nil {
			return nil, fmt.Errorf("AssetInfo %s can't read by error: %v", name, err)
		}
		return a.info, nil
	}
	return nil, fmt.Errorf("AssetInfo %s not found", name)
}

// AssetNames returns the names of the assets.
func AssetNames() []string {
	names := make([]string, 0, len(_bindata))
	for name := range _bindata {
		names = append(names, name)
	}
	return names
}

// _bindata is a table, holding each asset generator, mapped to its name.
var _bindata = map[string]func() (*asset, error){
	"Squeezenet.yml": squeezenetYml,
	"bvlc_googlenet.yml": bvlc_googlenetYml,
}

// AssetDir returns the file names below a certain
// directory embedded in the file by go-bindata.
// For example if you run go-bindata on data/... and data contains the
// following hierarchy:
//     data/
//       foo.txt
//       img/
//         a.png
//         b.png
// then AssetDir("data") would return []string{"foo.txt", "img"}
// AssetDir("data/img") would return []string{"a.png", "b.png"}
// AssetDir("foo.txt") and AssetDir("notexist") would return an error
// AssetDir("") will return []string{"data"}.
func AssetDir(name string) ([]string, error) {
	node := _bintree
	if len(name) != 0 {
		cannonicalName := strings.Replace(name, "\\", "/", -1)
		pathList := strings.Split(cannonicalName, "/")
		for _, p := range pathList {
			node = node.Children[p]
			if node == nil {
				return nil, fmt.Errorf("Asset %s not found", name)
			}
		}
	}
	if node.Func != nil {
		return nil, fmt.Errorf("Asset %s not found", name)
	}
	rv := make([]string, 0, len(node.Children))
	for childName := range node.Children {
		rv = append(rv, childName)
	}
	return rv, nil
}

type bintree struct {
	Func     func() (*asset, error)
	Children map[string]*bintree
}
var _bintree = &bintree{nil, map[string]*bintree{
	"Squeezenet.yml": &bintree{squeezenetYml, map[string]*bintree{}},
	"bvlc_googlenet.yml": &bintree{bvlc_googlenetYml, map[string]*bintree{}},
}}

// RestoreAsset restores an asset under the given directory
func RestoreAsset(dir, name string) error {
	data, err := Asset(name)
	if err != nil {
		return err
	}
	info, err := AssetInfo(name)
	if err != nil {
		return err
	}
	err = os.MkdirAll(_filePath(dir, filepath.Dir(name)), os.FileMode(0755))
	if err != nil {
		return err
	}
	err = ioutil.WriteFile(_filePath(dir, name), data, info.Mode())
	if err != nil {
		return err
	}
	err = os.Chtimes(_filePath(dir, name), info.ModTime(), info.ModTime())
	if err != nil {
		return err
	}
	return nil
}

// RestoreAssets restores an asset under the given directory recursively
func RestoreAssets(dir, name string) error {
	children, err := AssetDir(name)
	// File
	if err != nil {
		return RestoreAsset(dir, name)
	}
	// Dir
	for _, child := range children {
		err = RestoreAssets(dir, filepath.Join(name, child))
		if err != nil {
			return err
		}
	}
	return nil
}

func _filePath(dir, name string) string {
	cannonicalName := strings.Replace(name, "\\", "/", -1)
	return filepath.Join(append([]string{dir}, strings.Split(cannonicalName, "/")...)...)
}

