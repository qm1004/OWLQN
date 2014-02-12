package owlqn

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"strconv"
	"strings"
)

type LogisticRegression struct {
	indices         []int
	values          []float32
	instance_starts []uint32
	labels          []bool
	numFeats        int
}

func NewLogisticRegression(feature_file string, feature_num int) *LogisticRegression { //feature_file should be same format as libsvm's input file
	lr := new(LogisticRegression)
	lr.numFeats = feature_num
	f, err := os.Open(feature_file)
	if err != nil {
		fmt.Printf("%v\n", err)
		os.Exit(1)
	}
	//close file before return
	defer f.Close()

	//use bufio read file by row
	br := bufio.NewReader(f)
	row := 0
	lr.instance_starts = append(lr.instance_starts, 0)
	for {
		//read by row
		line, err := br.ReadString('\n')
		//fmt.Println(line,err)

		line = strings.TrimSpace(line)
		if len(line) > 0 {
			vs := strings.Split(line, "\t")
			label, _ := strconv.Atoi(vs[0])
			var bLabel bool
			switch label {
			case 1:
				bLabel = true
			case 0:
				bLabel = false
			default:
				fmt.Println("illegal label: must be 1 or 0")
				os.Exit(1)
			}
			lr.labels = append(lr.labels, bLabel)
			for i := 1; i < len(vs); i++ {
				kv := strings.Split(vs[i], ":")
				col, err := strconv.Atoi(kv[0])
				if err != nil {
					// handle error
					fmt.Println(err)
					os.Exit(2)
				}
				tempval, err := strconv.ParseFloat(kv[1], 32)
				if err != nil {
					// handle error
					fmt.Println(err)
					os.Exit(2)
				}
				lr.indices = append(lr.indices, col-1)
				lr.values = append(lr.values, float32(tempval))
			}
			lr.instance_starts = append(lr.instance_starts, uint32(len(lr.indices)))
			row++
		}
		if err == io.EOF {
			fmt.Printf("read %d rows\n", row)
			break
		}
	}
	fmt.Println("indices:", lr.indices)
	fmt.Println("values:", lr.values)
	fmt.Println("instance_starts:", lr.instance_starts)
	fmt.Println("labels:", lr.labels)

	return lr
}

func (lr *LogisticRegression) ScoreOf(i int, weights []float32) float32 {
	var score float32 = 0
	for j := lr.instance_starts[i]; j < lr.instance_starts[i+1]; j++ {
		var value float32 = lr.values[j]
		var index int = lr.indices[j]
		score += weights[index] * value
		if math.IsNaN(float64(weights[index])) {
			fmt.Println(weights, index, weights[index])
			os.Exit(1)
		}
	}
	if !lr.labels[i] {
		score *= -1
	}
	return score
}

func (lr *LogisticRegression) AddInstance(inds []int, vals []float32, label bool) {
	if len(inds) != len(vals) {
		log.Fatal("length of arrays are not same ")
	}
	for i := 0; i < len(inds); i++ {
		lr.indices = append(lr.indices, inds[i])
		lr.values = append(lr.values, vals[i])
	}
	lr.instance_starts = append(lr.instance_starts, uint32(len(lr.values)))
	lr.labels = append(lr.labels, label)
}
func (lr *LogisticRegression) AddMultTo(i int, mult float32, vec[]float32) {
	if lr.labels[i] {
		mult *= -1
	}
	for j := lr.instance_starts[i]; j < lr.instance_starts[i+1]; j++ {
		index := lr.indices[j]
		vec[index] += mult * lr.values[j]
	}
}

func (lr *LogisticRegression) NumInstances() int {
	return len(lr.labels)
}

type LogisticRegressionObjective struct {
	lr       *LogisticRegression
	l2weight float32
}

func NewLogisticRegressionObjective(logisticregression *LogisticRegression, l2 float32) *LogisticRegressionObjective {
	return &LogisticRegressionObjective{
		lr:       logisticregression,
		l2weight: l2,
	}
}
func (obj *LogisticRegressionObjective) Eval(input, gradient []float32) float32 {
	var loss float32
	loss = 1
	clear(gradient)
	if obj.l2weight != 0 {
		for i := 0; i < len(input); i++ {
			loss += input[i] * input[i] * obj.l2weight
			gradient[i] += obj.l2weight * input[i]
		}
		fmt.Println("l2weight is not zero!")
	}
	for i := 0; i < obj.lr.NumInstances(); i++ {
		score := obj.lr.ScoreOf(i, input)

		var insLoss, insProb float32
		if score < -30 {
			insLoss = -score
			insProb = 0
		} else if score > 30 {
			insLoss = 0
			insProb = 1
		} else {
			temp := 1.0 + math.Exp(float64(-score))
			insLoss = float32(math.Log(temp))
			insProb = 1.0 / float32(temp)
		}
		loss += insLoss
		obj.lr.AddMultTo(i, 1.0-insProb, gradient)
		/*temp:=1.0 + math.Exp(float64(-score))
		
		hx := 1.0 / float32(temp)
		if obj.lr.labels[i] {
			loss += float32(math.Log(temp))
			obj.lr.AddMultTo(i, hx-1.0, gradient)
		}else{
			loss += float32(-math.Log(1.0-float64(hx)))
			obj.lr.AddMultTo(i, hx, gradient)
		}*/
		
	}
	return loss
}
