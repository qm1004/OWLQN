package owlqn

import (
	//"fmt"
	"log"
)

func dotProduct(a, b []float32) float32 {
	if len(a) != len(b) {
		log.Fatal("length of arrays are not same ")
	}
	var result float32
	result = 0
	for i := 0; i < len(a); i++ {
		result += a[i] * b[i]
	}
	return result
}

func clear(a []float32) {
	for i := 0; i < len(a); i++ {
		a[i]=0
	}
}

func addMult(a, b []float32, c float32) {
	if len(a) != len(b) {
		log.Fatal("length of arrays are not same ")
	}
	for i := 0; i < len(a); i++ {
		a[i] += b[i] * c
	}
}

func add(a, b []float32) {
	if len(a) != len(b) {
		log.Fatal("length of arrays are not same ")
	}
	for i := 0; i < len(a); i++ {
		a[i] += b[i]
	}
}

func addMultInto(a, b, c []float32, d float32) {
	if len(a) != len(b) || len(a) != len(c) {
		log.Fatal("length of arrays are not same ")
	}
	for i := 0; i < len(a); i++ {
		a[i] = b[i] + c[i]*d
	}
}

func scale(a []float32, b float32) {
	for i := 0; i < len(a); i++ {
		a[i] *= b
	}
}

func scaleInto(a, b []float32, c float32) {
	if len(a) != len(b) {
		log.Fatal("length of arrays are not same ")
	}
	for i := 0; i < len(a); i++ {
		a[i] = b[i] * c
	}
}

func DeepCopy(a, b []float32) {
	if len(a) != len(b) {
		log.Fatal("length of arrays are not same ")
	}
	for i := 0; i < len(a); i++ {
		a[i] = b[i]
	}
}

func Mod(a, b int) int {
	if b <= 0 || a < 0 {
		log.Fatal("number illegal:should set a >=0 and b>0")
	}
	return a - a/b*b
}

func abs(a float32) float32 {
	if a < 0.0 {
		return -a
	} else {
		return a
	}
}
