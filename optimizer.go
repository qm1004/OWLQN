package owlqn

import (
	"log"
	"math"
	"fmt"
)

const  DEBUG bool=false
type costfunction interface {
	Eval(x, y []float32) float32
}

type OptimizerState struct {
	sList                       [][]float32 //s[i][k] = newX[k+1] - x[k] , m>i>=0
	yList                       [][]float32 //y[i][k] = newGrad[k+1]- grad[k]
	x, grad, newX, newGrad, dir []float32
	roList, alphas              []float32
	steepestDescDir             []float32
	value                       float32 //cost value
	m, iter                     int     //m:steps to save
	dim                         int     //feature number
	l1weight                    float32
	quiet                       bool

	costf costfunction
}

func New2dimSlice(steps, dimension int) [][]float32 {
	v := make([][]float32, steps)
	for i := 0; i < steps; i++ {
		v[i] = make([]float32, dimension)
	}
	return v
}

//init OptimizerState
func NewOptimizerState(f costfunction, steps int, init []float32, l1weight float32, quiet bool) *OptimizerState {
	state := new(OptimizerState)
	state.m = steps
	state.dim = len(init)
	state.iter = 0
	state.l1weight = l1weight
	state.quiet = quiet

	state.x = make([]float32, state.dim)
	state.grad = make([]float32, state.dim)
	state.newX = make([]float32, state.dim)
	state.newGrad = make([]float32, state.dim)
	state.dir = make([]float32, state.dim)
	state.steepestDescDir = state.newGrad //references newGrad to save memory, since we don't ever use both at the same time

	DeepCopy(state.x,init)
	DeepCopy(state.newX,init)

	state.alphas = make([]float32, state.m)
	state.roList = make([]float32, state.m)

	state.sList = New2dimSlice(state.m, state.dim)
	state.yList = New2dimSlice(state.m, state.dim)

	state.costf = f

	state.value = state.EvalL1()
	DeepCopy(state.grad,state.newGrad)
/*	fmt.Println("****************************")
	fmt.Println("newGrad:",state.newGrad)
	fmt.Println("grad:",state.grad)
	fmt.Println("newX:",state.newX)
	fmt.Println("x:",state.x)
	fmt.Println("****************************")*/
	return state
}

func (state *OptimizerState)GetValue() float32{
	return state.value
}

func (state *OptimizerState) EvalL1() float32 {
	var val float32
	//fmt.Println("iter:",state.iter)
	val = state.costf.Eval(state.newX, state.newGrad)
	//fmt.Println("EvalL1_val:",val)
	if state.l1weight > 0 {
		for i := 0; i < state.dim; i++ {
			val += abs(state.newX[i]) * state.l1weight
		}
	}

	return val
}
func (state *OptimizerState) UpdateDir() {
	//fmt.Println("state.iter:",state.iter)
	state.MakeSteepestDescDir()
	state.MapDirByInverseHessian()
	state.FixDirSigns()

	if DEBUG {
		state.TestDirDeriv()
	}

}



func (state *OptimizerState) TestDirDeriv() {
	dirNorm:=float32(math.Sqrt(float64(dotProduct(state.dir, state.dir))))
	var eps float32 =1.05e-8 / dirNorm
	//fmt.Println("eps:",eps)
	state.GetNextPoint(eps)
	val2:=state.EvalL1()
	fmt.Println("val2,value,eps:",float64(val2),float64(state.value),eps)
	numDeriv :=(val2 - state.value) / eps
	deriv:=state.DirDeriv()
	if !state.quiet {
		fmt.Printf("Grad check:%f vs. %f\n",numDeriv,deriv)
	}
	
}
func (state *OptimizerState) TestDirDeriv2() {
	var eplison float32 = 1e-4
	numdir:=make([]float32,len(state.dir))
	for i := 0; i < len(state.newX); i++ {
		state.newX[i]+=eplison
		val1:=state.EvalL1()
		state.newX[i]-=2*eplison
		val2:=state.EvalL1()
		state.newX[i]+=eplison
		numdir[i]=-(val1-val2)/(2*eplison)
	}
	state.EvalL1()
	/*fmt.Println("numdir:",numdir)
	fmt.Println("   dir:",state.dir)
	fmt.Println("grad:",state.grad)*/
	var x float32=0
	var y float32=0
	for i := 0; i < len(numdir); i++ {
		x+=(numdir[i]-state.dir[i])*(numdir[i]-state.dir[i])
		y+=(numdir[i]+state.dir[i])*(numdir[i]+state.dir[i])
	}
	diff:=math.Sqrt(float64(x))/math.Sqrt(float64(y))
	fmt.Println("diff:",diff,"numdir:",numdir,"state.dir:",state.dir)
	if diff >1e-3 {
		fmt.Println("dir diff > 1e-3")
	}
}

func (state *OptimizerState) MakeSteepestDescDir() {
	if state.l1weight == 0 {
		scaleInto(state.dir, state.grad, -1)
		fmt.Println("l1weight=0.0 MakeSteepestDescDir state.dir state.grad:",state.dir,state.grad)
	} else {
		for i := 0; i < state.dim; i++ {
			if state.x[i] < 0 {
				state.dir[i] = -state.grad[i] + state.l1weight
			} else if state.x[i] > 0 {
				state.dir[i] = -state.grad[i] - state.l1weight
			} else {
				if state.grad[i] < -state.l1weight {
					state.dir[i] = -state.grad[i] - state.l1weight
				} else if state.grad[i] > state.l1weight {
					state.dir[i] = -state.grad[i] + state.l1weight
				} else {
					state.dir[i] = 0
				}

			}

		}
		//fmt.Println("MakeSteepestDescDir dir:",state.dir," MakeSteepestDescDir state.grad:",state.grad)
	}
	DeepCopy(state.steepestDescDir,state.dir)
}

func (state *OptimizerState) MapDirByInverseHessian() {
	if state.iter == 0 {
		fmt.Println("iter=0 return now!!")
		return
	}
	lowerBound := state.iter - state.m
	if lowerBound < 0 {
		lowerBound = 0
	}
	//这部分有错
	for i := state.iter - 1; i >= lowerBound; i-- {
		currIndex := Mod(i, state.m)
		state.alphas[currIndex] = -dotProduct(state.sList[currIndex], state.dir) / state.roList[currIndex]
		addMult(state.dir, state.yList[currIndex], state.alphas[currIndex])
	}
	prevIndex := Mod(state.iter-1, state.m)
	lastY := state.yList[prevIndex]
	yDotY := dotProduct(lastY, lastY)
	scalar := state.roList[prevIndex] / yDotY
	scale(state.dir, scalar)

	for i := lowerBound; i <=state.iter-1; i++ {
		currIndex := Mod(i, state.m)
		//fmt.Println("currIndex:",currIndex)
		beta := dotProduct(state.yList[currIndex], state.dir)/state.roList[currIndex]
		addMult(state.dir, state.sList[currIndex], -state.alphas[currIndex]-beta)
	}
	/*fmt.Println("lowerBound:",lowerBound)
	fmt.Println("MapDirByInverseHessian dir:",state.dir)
	fmt.Println("state.yList:",state.yList)
	fmt.Println("state.sList:",state.sList)
	fmt.Println("state.roList:",state.roList)
	fmt.Println("state.alphas:",state.alphas)
	fmt.Println("MapDirByInverseHessian state.dir state.grad:",state.dir,state.grad)*/
}

func (state *OptimizerState) FixDirSigns() {
	if state.l1weight > 0 {
		for i := 0; i < state.dim; i++ {
			if state.dir[i]*state.steepestDescDir[i] <= 0 {
				state.dir[i] = 0
			}
		}
	}
}

func (state *OptimizerState) DirDeriv() float32 {
	if state.l1weight == 0 {
		return dotProduct(state.dir, state.grad)
	} else {
		var val float32
		for i := 0; i < state.dim; i++ {
			if state.dir[i] != 0 {
				if state.x[i] < 0 {
					val += state.dir[i] * (state.grad[i] - state.l1weight) //search direction multiply by gradient when x!=0
				} else if state.x[i] > 0 {
					val += state.dir[i] * (state.grad[i] + state.l1weight)
				} else if state.dir[i] < 0 {
					val += state.dir[i] * (state.grad[i] - state.l1weight) // search direction multiply by pseudo-gradient when x=0
				} else if state.dir[i] > 0 {
					val += state.dir[i] * (state.grad[i] + state.l1weight)
				}
			}
			//fmt.Println(i,state.dir[i],state.x[i],state.grad[i],state.l1weight)
		}
		return val
	}
}

func (state *OptimizerState) GetNextPoint(alpha float32) {
	addMultInto(state.newX, state.x, state.dir, alpha)
	if state.l1weight > 0 {
		for i := 0; i < state.dim; i++ {
			if state.x[i]*state.newX[i] < 0.0 {
				state.newX[i] = 0.0
			}
		}
	}
}

func (state *OptimizerState) BackTrackingLineSearch() {
	origDirDeriv := state.DirDeriv()
	// if a non-descent direction is chosen, the line search will break anyway, so throw here
	// The most likely reason for this is a bug in your function's gradient computation
	if origDirDeriv >= 0 {
		fmt.Println("origDirDeriv:",origDirDeriv,state.dir,state.grad)
		log.Fatal("L-BFGS chose a non-descent direction: check your gradient!")
	}
	var alpha float32 = 1.0
	var backoff float32 = 0.5
	if state.iter == 0 {
		var normDir float32 = float32(math.Sqrt(float64(dotProduct(state.dir, state.dir))))
		alpha = 1 / normDir
		//fmt.Println("alpha:",alpha)
		backoff = 0.1
	}
	const c1 float32 = 0.0001
	var oldValue = state.value
	k:=0
	for true {
		state.GetNextPoint(alpha)
		state.value = state.EvalL1()

		if state.value <= oldValue+c1*origDirDeriv*alpha {
			//fmt.Println("k:",k)
			break
		}
		alpha *= backoff
		k++
		
	}
}

func (state *OptimizerState) Shift() {

	nextS := make([]float32, state.dim)
	nextY := make([]float32, state.dim)
	addMultInto(nextS, state.newX, state.x, -1)
	addMultInto(nextY, state.newGrad, state.grad, -1)    

	var ro float32 = dotProduct(nextS, nextY)

	var currIndex int = Mod(state.iter, state.m)
	DeepCopy(state.sList[currIndex],nextS)
	DeepCopy(state.yList[currIndex],nextY)
	state.roList[currIndex] = ro

	DeepCopy(state.x,state.newX)
	DeepCopy(state.grad,state.newGrad)

	state.iter++

}
