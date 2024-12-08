package leetcode

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
)

// lengthOfLongestSubstring 给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。
func lengthOfLongestSubstring(s string) int {
	l, r := 0, 0
	rs := []rune(s)
	m := make(map[rune]int)
	var res int
	for r < len(rs) {
		rn := rs[r]
		m[rn]++
		r++
		for m[rn] > 1 {
			m[rs[l]]--
			l++
		}
		if r-l > res {
			res = r - l
		}
	}
	return res
}

/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
//给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。
func reverseList(head *ListNode) *ListNode {
	var prev *ListNode
	curr := head
	for curr != nil {
		next := curr.Next
		curr.Next = prev
		prev = curr
		curr = next
	}
	return prev
}

type KV struct {
	key int
	val int
}

// 请你设计并实现一个满足  LRU (最近最少使用) 缓存 约束的数据结构。
/**
 * Your LRUCache object will be instantiated and called as such:
 * obj := Constructor(capacity);
 * param_1 := obj.Get(key);
 * obj.Put(key,value);
 */

type LRUCache struct {
	m        map[int]*DqueueNode
	capacity int
	dq       *Dqueue
}

func Constructor(capacity int) LRUCache {
	return LRUCache{
		m:        make(map[int]*DqueueNode),
		dq:       NewDqueue(),
		capacity: capacity,
	}
}

func (this *LRUCache) Get(key int) int {
	if val, ok := this.m[key]; ok {
		this.dq.MoveToLast(val)
		return val.Val.(*KV).val
	}
	return -1
}

func (this *LRUCache) Put(key int, value int) {
	if dqueueNode, ok := this.m[key]; ok {
		kv := dqueueNode.Val.(*KV)
		kv.val = value
		this.dq.MoveToLast(dqueueNode)
		return
	}

	node := &DqueueNode{Val: &KV{key: key, val: value}}
	this.m[key] = node
	this.dq.AddLast(node)

	if this.capacity < len(this.m) {
		delNode := this.dq.Head.Next
		val := delNode.Val.(*KV)
		delete(this.m, val.key)
		this.dq.Remove(delNode)
	}
}

func findKthLargest(nums []int, k int) int {
	var qselect func(l, r, k int) int
	qselect = func(l, r, k int) int {
		if l == r {
			return nums[k]
		}
		partition := nums[l]
		i := l
		j := r
		for i < j {
			for ; nums[i] < partition; i++ {
			}
			for ; nums[j] > partition; j-- {
			}
			if i < j {
				nums[i], nums[j] = nums[j], nums[i]
			}
		}
		if k <= j {
			return qselect(l, j, k)
		} else {
			return qselect(j+1, r, k)
		}

	}

	return qselect(0, len(nums)-1, len(nums)-k)
}

func Max(arr []int) int {
	if len(arr) == 0 {
		panic("invalid slice len")
	}
	if len(arr) == 1 {
		return arr[0]
	}

	max := Max(arr[1:])
	if arr[0] > max {
		return arr[0]
	} else {
		return max
	}
}

func threeSum(nums []int) [][]int {
	sort.Slice(nums, func(i, j int) bool {
		return nums[i] < nums[j]
	})
	idxMap := make(map[int]int)
	for i, num := range nums {
		idxMap[num] = i
	}
	var res [][]int
	for i := 0; i < len(nums)-1; i++ {
		if i > 0 && nums[i] == nums[i-1] {
			continue
		}
		num1 := nums[i]
		for j := i + 1; j < len(nums); j++ {
			if j > i+1 && nums[j] == nums[j-1] {
				continue
			}
			sumTwo := num1 + nums[j]
			idx, ok := idxMap[-sumTwo]
			if ok && idx > j {
				res = append(res, []int{num1, nums[j], nums[idx]})
			}
		}
	}
	return res
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func levelOrderBottom(root *TreeNode) [][]int {
	var res [][]int
	if root == nil {
		return res
	}
	var nodes []*TreeNode
	nodes = append(nodes, root)
	for len(nodes) != 0 {
		var children []*TreeNode
		var currentLevelVals []int
		for _, node := range nodes {
			currentLevelVals = append(currentLevelVals, node.Val)
			if node.Left != nil {
				children = append(children, node.Left)
			}
			if node.Right != nil {
				children = append(children, node.Right)
			}
		}
		res = append(res, currentLevelVals)
		nodes = children
	}
	for i := 0; i < len(res)/2; i++ {
		res[i], res[len(res)-i-1] = res[len(res)-i-1], res[i]
	}
	return res
}

func zigzagLevelOrder(root *TreeNode) [][]int {
	var res [][]int
	if root == nil {
		return res
	}
	nodes := []*TreeNode{root}
	var reverse bool
	for len(nodes) > 0 {
		var children []*TreeNode
		var levelVals []int
		for i := 0; i < len(nodes); i++ {
			node := nodes[i]
			levelVals = append(levelVals, node.Val)
			if node.Left != nil {
				children = append(children, node.Left)
			}
			if node.Right != nil {
				children = append(children, node.Right)
			}
		}
		if reverse {
			l := len(levelVals)
			for i := 0; i < l/2; i++ {
				levelVals[i], levelVals[l-1-i] = levelVals[l-1-i], levelVals[i]
			}
		}
		nodes = children
		res = append(res, levelVals)
		reverse = !reverse
	}
	return res
}

func mergeAlternately(word1 string, word2 string) string {
	rs1 := []rune(word1)
	rs2 := []rune(word2)
	n := len(rs1)
	var suffix string
	if len(rs1) > len(rs2) {
		n = len(rs2)
		suffix = string(rs1[n:])
	} else if len(rs2) > len(rs1) {
		n = len(rs1)
		suffix = string(rs2[n:])
	}
	var sb strings.Builder
	for i := 0; i < n; i++ {
		sb.WriteRune(rs1[i])
		sb.WriteRune(rs2[i])
	}
	return sb.String() + suffix
}

func gcdOfStrings(str1 string, str2 string) string {
	var m sync.Map
	m.Load("aaa")
	m.Store("aaa", "aa")
	canDivide := func(str string, subStr string) bool {
		sLen := len(subStr)
		if len(str)%sLen != 0 {
			return false
		}
		for i := 0; i < len(str); i++ {
			if str[i] != subStr[i%sLen] {
				return false
			}
		}
		return true
	}

	var res string
	for i := 0; i < len(str1) && i < len(str2); i++ {
		s := str1[:i+1]
		if canDivide(str1, s) && canDivide(str2, s) {
			res = s
		}
	}
	return res
}

func buildTree(preorder []int, inorder []int) *TreeNode {
	idxMap := make(map[int]int)
	for i, v := range preorder {
		idxMap[v] = i
	}
	var buildNode func(inorder []int) *TreeNode
	buildNode = func(inorder []int) *TreeNode {
		if len(inorder) == 0 {
			return nil
		}

		var val int
		maxIdx := math.MaxInt
		for _, v := range inorder {
			if idxMap[v] < maxIdx {
				maxIdx = idxMap[v]
				val = v
			}
		}

		node := &TreeNode{Val: val}
		idx := -1
		for i, v := range inorder {
			if v == val {
				idx = i
			}
		}
		if idx > 0 {
			node.Left = buildNode(inorder[:idx])
		}
		if len(preorder) >= 2 && idx < len(inorder) {
			node.Right = buildNode(inorder[idx+1:])
		}
		return node
	}
	return buildNode(inorder)
}

func constructFromPrePost(preorder []int, postorder []int) *TreeNode {
	var dfs func(preorder []int, postorder []int) *TreeNode
	dfs = func(preorder []int, postorder []int) *TreeNode {
		if len(preorder) == 0 {
			return nil
		}
		postIdx := make(map[int]int)
		for i, v := range postorder {
			postIdx[v] = i
		}
		val := preorder[0]
		n := &TreeNode{
			Val: val,
		}
		if len(preorder) > 1 {
			leftVal := preorder[1]
			leftIdx := postIdx[leftVal]
			n.Left = dfs(preorder[1:leftIdx+2], postorder[:leftIdx+2])
			if leftIdx < len(preorder) {
				n.Right = dfs(preorder[leftIdx+2:], postorder[leftIdx+1:len(postorder)-1])
			}
		}
		return n
	}

	return dfs(preorder, postorder)
}

func kthLargestLevelSum(root *TreeNode, k int) int64 {
	var levelSumList []int64
	if root == nil {
		return -1
	}
	var q []*TreeNode
	q = append(q, root)
	for len(q) > 0 {
		levelSum, size := int64(0), len(q)
		for i := 0; i < size; i++ {
			node := q[0]
			q = q[1:]
			levelSum += int64(node.Val)
			if node.Left != nil {
				q = append(q, node.Left)
			}
			if node.Right != nil {
				q = append(q, node.Right)
			}
		}
		levelSumList = append(levelSumList, levelSum)
	}
	if k > len(levelSumList) {
		return -1
	}
	sort.Slice(levelSumList, func(i, j int) bool {
		return levelSumList[i] > levelSumList[j]
	})
	return levelSumList[k-1]
}

func closestNodes(root *TreeNode, queries []int) [][]int {
	var res [][]int
	if root == nil {
		return res
	}
	var arr []int
	var dfs func(node *TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		dfs(node.Left)
		arr = append(arr, node.Val)
		dfs(node.Right)
	}
	dfs(root)

	//ans := make([][]int, len(queries))
	//for i, q := range queries {
	//	mn, mx := -1, -1
	//	j, ok := slices.BinarySearch(arr, q)
	//	if j < len(arr) {
	//		mx = arr[j]
	//	}
	//	if !ok { // a[j]>q, a[j-1]<q
	//		j--
	//	}
	//	if j >= 0 {
	//		mn = arr[j]
	//	}
	//	ans[i] = []int{mn, mx}
	//}
	return res
}

func rangeSumBST(root *TreeNode, low int, high int) int {
	var res int
	var dfs func(node *TreeNode)
	dfs = func(node *TreeNode) {
		if node == nil {
			return
		}
		if node.Left != nil && node.Left.Val >= low {
			dfs(node.Left)
		}
		if node.Right != nil && node.Right.Val <= high {
			dfs(node.Right)
		}
		if node.Val >= low && node.Val <= high {
			res += node.Val
		}
	}
	dfs(root)
	return res
}

func minIncrements(n int, cost []int) int {
	if n <= 0 {
		return 0
	}
	abs := func(x int) int {
		if x < 0 {
			return -x
		}
		return x
	}
	max := func(a, b int) int {
		if a > b {
			return a
		}
		return b
	}

	var res int
	for i := n - 2; i > 0; i -= 2 {
		res += abs(cost[i] - cost[i+1])
		// 叶节点 i 和 i+1 的双亲节点下标为 i/2（整数除法）
		cost[i/2] = cost[i/2] + max(cost[i], cost[i+1])
	}
	return res
}

type Stack[T int | int64] struct {
	vals []T
}

func (s *Stack[T]) Pop() T {
	l := len(s.vals)
	if l == 0 {
		return 0
	}
	val := s.vals[l-1]
	s.vals = s.vals[:l-1]
	return val
}

func (s *Stack[T]) Push(val T) {
	s.vals = append(s.vals, val)
}

func (s *Stack[T]) Size() int {
	return len(s.vals)
}

func (s *Stack[T]) Empty() bool {
	return s.Size() == 0
}

func (s *Stack[T]) Peek() T {
	if s.Empty() {
		return 0
	}
	return s.vals[s.Size()-1]
}

type MyQueue struct {
	s1 *Stack[int]
	s2 *Stack[int]
}

func (this *MyQueue) Push(x int) {
	this.s2 = &Stack[int]{}
	this.s1.Push(x)
}

func (this *MyQueue) Pop() int {
	if !this.s2.Empty() {
		return this.s2.Pop()
	}
	for !this.s1.Empty() {
		this.s2.Push(this.s1.Pop())
	}
	return this.s2.Pop()
}

func (this *MyQueue) Empty() bool {
	return this.s1.Empty() && this.s2.Empty()
}

func (this *MyQueue) Peek() int {
	if !this.s2.Empty() {
		return this.s2.Peek()
	}
	for !this.s1.Empty() {
		this.s2.Push(this.s1.Pop())
	}
	return this.s2.Peek()
}

func findKOr(nums []int, k int) int {
	var ans int
	for i := 0; i < 31; i++ {
		cnt := 0
		for _, num := range nums {
			if (num>>i)&1 == 1 {
				cnt++
			}
		}
		if cnt >= k {
			ans |= 1 << i
		}
	}
	return ans
}

func divisibilityArray(word string, m int) []int {
	var ans []int
	var sum int
	for _, c := range word {
		sum = sum*10 + int(c-'0')
		if sum%m == 0 {
			ans = append(ans, 1)
		} else {
			ans = append(ans, 0)
		}
	}
	return ans
}

func capitalizeTitle(title string) string {
	splits := strings.Split(title, " ")
	var res []string
	for _, split := range splits {
		split = strings.ToLower(split)
		if len(split) > 2 {
			split = strings.ToUpper(split[0:1]) + split[1:]
		}
		res = append(res, split)
	}
	return strings.Join(res, " ")
}

func maximumOddBinaryNumber(s string) string {
	cnt := strings.Count(s, "1")
	return strings.Repeat("1", cnt-1) + strings.Repeat("0", len(s)-cnt) + "1"
}

func maxArrayValue(nums []int) int64 {
	l := len(nums)
	cpArr := make([]int64, l)
	for i, num := range nums {
		cpArr[i] = int64(num)
	}
	for i := l - 1; i > 0; i-- {
		if cpArr[i] >= cpArr[i-1] {
			cpArr[i-1] += cpArr[i]
			cpArr[i] = -1
		}
	}
	var res int64
	for _, num := range cpArr {
		if num > res {
			res = num
		}
	}
	return res
}

func coinChange(coins []int, amount int) int {
	min := func(a, b int) int {
		if a < b {
			return a
		}
		return b
	}
	dp := make([]int, amount+1)
	for i := range dp {
		dp[i] = amount + 1
	}
	dp[0] = 0
	for i := 1; i <= amount; i++ {
		for _, coin := range coins {
			if coin <= i {
				dp[i] = min(dp[i], dp[i-coin]+1)
			}
		}
	}
	if dp[amount] > amount {
		return -1
	}
	return dp[amount]
}

func change(amount int, coins []int) int {
	// f(n) =
	dp := make([]int, amount+1)
	dp[0] = 1
	for i := 1; i <= amount; i++ {
		for _, coin := range coins {
			if i >= coin && dp[i-coin] > 0 {
				dp[i] += dp[i-coin]
			}
		}
	}
	fmt.Println(dp)
	return dp[amount]
}

func firstDayBeenInAllRooms(nextVisit []int) int {
	// f(n) = 1 +f(nextVisit[i])
	return 1
}

func maxAncestorDiff(root *TreeNode) int {
	var res int
	abs := func(a int) int {
		if a > 0 {
			return a
		}
		return -a
	}
	var dfs func(node *TreeNode, max int, min int)
	dfs = func(node *TreeNode, max int, min int) {
		if node == nil {
			return
		}
		i := abs(node.Val - min)
		if i > res {
			res = i
		}
		j := abs(node.Val - max)
		if j > res {
			res = j
		}
		if node.Val > max {
			max = node.Val
		}
		if node.Val < min {
			min = node.Val
		}

		dfs(node.Left, max, min)
		dfs(node.Right, max, min)
	}
	dfs(root, root.Val, root.Val)
	return res

}

func wateringPlants(plants []int, capacity int) int {
	water := capacity
	var res int
	for i, plant := range plants {
		if water < plant {
			res += 2*i + 1
			water = capacity - plant
		} else {
			res += 1
			water = water - plant
		}
	}
	return res
}

func minimumRefill(plants []int, capacityA int, capacityB int) int {
	l, r := 0, len(plants)-1
	var res int
	waterA := capacityA
	waterB := capacityB
	for l <= r {
		if l == r {
			if waterA < plants[l] && waterB < plants[l] {
				res++
			}
			l++
			r--
			continue
		}
		if waterA < plants[l] {
			res++
			waterA = capacityA - plants[l]
		} else {
			waterA -= plants[l]
		}
		l++

		if waterB < plants[r] {
			res++
			waterB = capacityB - plants[r]
		} else {
			waterB -= plants[r]
		}
		r--
	}
	return res
}

func garbageCollection(garbage []string, travel []int) int {
	cs := []int32{'M', 'P', 'G'}
	for i, j := range travel[1:] {
		travel[i+1] = travel[i] + j
	}

	var res int
	for _, c := range cs {
	Out:
		for j := len(garbage) - 1; j >= 0; j-- {
			str := garbage[j]
			for _, u := range str {
				if u == c {
					if j > 0 {
						res += travel[j-1]
					}
					break Out
				}
			}
		}
	}

	for _, str := range garbage {
		res += len(str)
	}

	return res
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func minDays(n int) int {
	if n <= 1 {
		return 1
	}
	if n%6 == 0 {
		return min(min(minDays(n-1), minDays(n/2)), minDays(n-2*(n/3))) + 1
	}
	if n%2 == 0 {
		return min(minDays(n-1), minDays(n/2)) + 1
	}
	if n%3 == 0 {
		return min(minDays(n-1), minDays(n-2*(n/3))) + 1
	}
	return minDays(n-1) + 1
}

func sumOfTheDigitsOfHarshadNumber(x int) int {
	var i int
	y := x
	for y > 0 {
		i += y % 10
		y = y / 10
	}
	if x%i == 0 {
		return i
	}
	return -1
}
