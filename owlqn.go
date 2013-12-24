package owlqn

import (
	"container/list"
	"fmt"
	"math"
)

const numItersToAvg int = 5

type TerminationCriterion struct {
	prevVals *list.List
}

func (termCrit *TerminationCriterion) GetValue(state *OptimizerState) float32 {
	var retval float32 = math.MaxFloat32
	if termCrit.prevVals.Len() >= numItersToAvg {
		var prevVal float32 = termCrit.prevVals.Front().Value.(float32)
		if termCrit.prevVals.Len() == 10 {
			firstContainerItem := termCrit.prevVals.Front()
			if firstContainerItem != nil {
				_ = termCrit.prevVals.Remove(firstContainerItem)
			}
		}
			var averageImprovement float32 = (prevVal - state.GetValue()) / float32(termCrit.prevVals.Len())
			var relAvgImpr float32 = averageImprovement / abs(state.GetValue())
			fmt.Printf("relAvgImpr:%f\n", relAvgImpr)
			retval = relAvgImpr
		} else {
			fmt.Println(" (wait for five iters) ")
		}
	
	termCrit.prevVals.PushBack(state.GetValue())
	return retval
}

type OWLQN struct {
	quiet    bool
	termCrit *TerminationCriterion
}

func NewTerminationCriterion() *TerminationCriterion {
	return &TerminationCriterion{
		prevVals: list.New(),
	}
}

func NewOWLQN(quiet bool) *OWLQN {
	return &OWLQN{
		quiet:    quiet,
		termCrit: NewTerminationCriterion(),
	}
}

func (opt *OWLQN) Minimize(f costfunction, init []float32, result []float32, l1weight float32, tol float32, m int) {
	state := NewOptimizerState(f, m, init, l1weight, opt.quiet)

	//fmt.Println("test:",state.x,state.grad,state.newGrad)
	
	if !opt.quiet {
		fmt.Printf("Optimizing function of %d variables with OWLQN parameters:\n", state.dim)
		fmt.Printf("l1 regularization weight:%f.\n", l1weight)
		fmt.Printf("L-BFGS memory parameter (m):%d\n", m)
		fmt.Printf("Convergence tolerance:%f\n", tol)
		fmt.Printf("Iter   n:    new_value\n")
		fmt.Printf("Iter   0:    %f\n", state.value)
		}
		for true {
			if !opt.quiet {
				fmt.Printf("Iter   %d\n", state.iter)
				
			}
			state.UpdateDir()
			//fmt.Println("BackTrackingLineSearch")
			//fmt.Println("dir:",state.dir)
			//fmt.Println("grad:",state.grad)
			//fmt.Println("newX:",state.newX)
			state.BackTrackingLineSearch()

			var termCritVal float32 = opt.termCrit.GetValue(state)
			if !opt.quiet {
				//fmt.Printf("Iter   %d:    %f\n", state.iter, state.value)
				fmt.Printf("state.value:%f\n",  state.value)
				
			}
			if termCritVal < tol {
				break
			}
			state.Shift()
		}
		DeepCopy(result,state.newX)
	
}
