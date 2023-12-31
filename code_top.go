package leetcode

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
