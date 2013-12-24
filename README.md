OWLQN
=====
#Introduction
OWLQN is an optimization algorithm,which is called L1-LBFGS.
OWLQN algorithm is proposed by paper "Orthant-Wise Limited-memory Quasi-Newton Optimizer for L1-regularized Objectives".

This paper has provided a c++ source code running Windows.you can downlaod it at http://research.microsoft.com/en-us/downloads/b1eb1016-1738-4bd5-83a9-370c9d498a03/
Most of my go code refer to c++ source code,but I think there is some little bug I fixed in go code.

#How to use
```go
package main

import (
    "bufio"
    "flag"
    "fmt"
    "github.com/qm1004/OWLQN"
    "os"
    "strconv"
    "math/rand"
)

func main() {
    var l1weight float64
    var l2weight float64
    var feature_num int
    var m int
    var tol float64
    var quiet bool
    var feature_file string
    var output_file string
    flag.Float64Var(&l1weight, "l1weight", 1.0, "coefficient of l1 regularizer (default is 1)")
    flag.Float64Var(&l2weight, "l2weight", 0.0, "coefficient of l2 regularizer(default is 0)")
    flag.IntVar(&m, "m", 10, "sets L-BFGS memory parameter (default is 10)")
    flag.IntVar(&feature_num, "num", 0, "total feature count")
    flag.Float64Var(&tol, "tol", 0.0001, " sets convergence tolerance (default is 1e-4)")
    flag.BoolVar(&quiet, "quiet", false, "Suppress all output information")
    flag.StringVar(&feature_file, "input", "./testdata", "the path of input feature file")
    flag.StringVar(&output_file, "output", "./model", "the path of output model file")
    flag.Parse()
    fmt.Println("feature_num:",feature_num,"  l1weight:",l1weight)
    
    lr:= owlqn.NewLogisticRegression(feature_file, feature_num)
    obj := owlqn.NewLogisticRegressionObjective(lr, float32(l2weight))
    init := make([]float32, feature_num)

    for i := 0; i < len(init); i++ {
        init[i]=rand.Float32()
    }

    result := make([]float32, feature_num)
    opt := owlqn.NewOWLQN(quiet)
    fmt.Println("init:",init)
    opt.Minimize(obj, init, result, float32(l1weight), float32(tol), m)

    nonZero := 0
    for i := 0; i < feature_num; i++ {
        if result[i] != 0.0 {
            nonZero++
        }
    }
    fmt.Printf("Finished train,%d/%d nonZero weight\n", nonZero, feature_num)
    fmt.Println("result:",result)

    f, err := os.Create(output_file)
    if err != nil {
        fmt.Printf("%v\n", err)
        os.Exit(1)
    }
    defer f.Close()
    br := bufio.NewWriter(f)
    br.WriteString("feature_num="+strconv.FormatInt(int64(feature_num), 10)+"\n")
    br.WriteString("l1weight="+strconv.FormatFloat(float64(l1weight), 'f', 4, 32)+"\n")
    br.Flush()
    for i := 0; i < feature_num; i++ {
        res := strconv.FormatFloat(float64(result[i]), 'f', 4, 32)
        br.WriteString(res + "\n")
        br.Flush()
    }

}
```

