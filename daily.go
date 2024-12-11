package leetcode

// leetcode  999
func numRookCaptures(board [][]byte) int {
	// 找到白车的位置
	wCar := byte('R')
	wElephant := byte('B')
	bSoldier := byte('p')
	var i int
	var j int
OUT:
	for ; i < 8; i++ {
		j = 0
		for ; j < 8; j++ {
			if board[i][j] == wCar {
				break OUT
			}
		}
	}

	// 4个方向前进看是否能吃到的黑色士兵
	dx := []int{1, -1, 0, 0}
	dy := []int{0, 0, -1, 1}
	var res int
	for k := 0; k < 4; k++ {
		for step := 0; ; step++ {
			tx, ty := i+dx[k]*step, j+dy[k]*step
			if tx >= 8 || ty >= 8 || tx < 0 || ty < 0 || board[tx][ty] == wElephant {
			}
			if board[tx][ty] == bSoldier {
				res++
				break
			}
		}
	}
	return res
}

// 2717
func semiOrderedPermutation(nums []int) int {
	n := len(nums)
	if n == 0 {
		return 0
	}
	if nums[0] == 1 && nums[n-1] == n {
		return 0
	}
	idx1 := -1
	idxn := -1
	for i, v := range nums {
		if v == 1 {
			idx1 = i
		}
		if v == n {
			idxn = i
		}
		if idx1 != -1 && idxn != -1 {
			break
		}
	}
	if idxn > idx1 {
		return idx1 + n - (idxn + 1)
	}
	//  idxn 小于idx1 的情况，先移动n 或 1 都会使另一个数字交换一个位置
	return idx1 + n - (idxn + 1) - 1
}
