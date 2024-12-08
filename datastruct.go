package leetcode

type ListNode struct {
	Val  int
	Next *ListNode
}

type Dqueue struct {
	Head *DqueueNode
	Tail *DqueueNode
}

func NewDqueue() *Dqueue {
	d := &Dqueue{}
	d.Head = &DqueueNode{}
	d.Tail = &DqueueNode{}
	d.Tail.Prev = d.Head
	d.Head.Next = d.Tail
	return d
}

func (d *Dqueue) Remove(node *DqueueNode) {
	if node == nil {
		return
	}
	next := node.Next
	prev := node.Prev
	prev.Next = next
	next.Prev = prev
}

func (d *Dqueue) AddLast(node *DqueueNode) {
	tail := d.Tail
	prev := tail.Prev
	prev.Next = node
	tail.Prev = node
	node.Next = tail
	node.Prev = prev
}

func (d *Dqueue) MoveToLast(node *DqueueNode) {
	d.Remove(node)
	d.AddLast(node)
}

type DqueueNode struct {
	Val  any
	Next *DqueueNode
	Prev *DqueueNode
}
