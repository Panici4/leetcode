package leetcode

import (
	"fmt"
	"testing"
)

func TestLruCache(t *testing.T) {
	lruCache := Constructor(2)
	lruCache.Put(1, 1)
	lruCache.Put(2, 2)
	lruCache.Get(1)
	lruCache.Put(3, 3)

}

func TestFindKthLargest(t *testing.T) {
	fmt.Println(Max([]int{5, 1, 2}))
}

func TestLock(t *testing.T) {

	root := &TreeNode{
		Val: 5,
		Left: &TreeNode{
			Val: 8,
			Left: &TreeNode{
				Val: 2,
				Left: &TreeNode{
					Val: 4,
				},
				Right: &TreeNode{
					Val: 6,
				},
			},
			Right: &TreeNode{
				Val: 1,
			},
		},
		Right: &TreeNode{
			Val: 9,
			Left: &TreeNode{
				Val: 3,
			},
			Right: &TreeNode{
				Val: 7,
			},
		},
	}
	kthLargestLevelSum(root, 2)

}

func TestChange(t *testing.T) {
	change(5, []int{1, 2, 5})
}
