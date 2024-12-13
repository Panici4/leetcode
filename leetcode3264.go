package leetcode

import "container/heap"

// 3264
func getFinalState(nums []int, k int, multiplier int) []int {
	h := new(arr)
	for i, v := range nums {
		h.Push([2]int{v, i})
	}
	heap.Init(h)
	for i := 0; i < k; i++ {
		pop := heap.Pop(h).([2]int)
		pop[0] = pop[0] * multiplier
		heap.Push(h, pop)
	}

	heapArr := *h
	res := make([]int, len(heapArr))
	for i := 0; i < len(heapArr); i++ {
		v := heapArr[i]
		res[v[1]] = v[0]
	}
	return res
}

type arr [][2]int

func (a *arr) Len() int {
	return len(*a)
}

func (a *arr) Less(i, j int) bool {
	v1 := (*a)[i]
	v2 := (*a)[j]
	if v1[0] == v2[0] {
		return v1[1] < v2[1]
	}
	return v1[0] < v2[0]
}

func (a *arr) Swap(i, j int) {
	(*a)[i], (*a)[j] = (*a)[j], (*a)[i]
}

func (a *arr) Push(x any) {
	*a = append(*a, x.([2]int))
}

func (a *arr) Pop() (v any) {
	*a, v = (*a)[:a.Len()-1], (*a)[a.Len()-1]
	return
}
