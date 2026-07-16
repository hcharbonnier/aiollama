package sdcpp

/*
#cgo CFLAGS: -O3 -I${SRCDIR}/include
#include "stable-diffusion.h"
#include <stdlib.h>
*/
import "C"

import (
	"unsafe"
)

type testSDImage = C.sd_image_t

func testCImageToGo(img *C.sd_image_t) Image {
	return cImageToGo(img)
}

func testGoImageToCAndBack(img *Image) Image {
	cImg := goImageToC(img)
	defer func() {
		if cImg.data != nil {
			C.free(unsafe.Pointer(cImg.data))
		}
	}()
	return cImageToGo(&cImg)
}
