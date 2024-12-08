package leetcode

// leetcode  999
func numRookCaptures(board [][]byte) int {
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
