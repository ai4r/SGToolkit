class Node {
    constructor(item, prev, next) {
        this.item = item;
        this.prev = prev;
        this.next = next;
    }
}

class SortedLinkedList {
    constructor(sortAttr) {
        this.head = new Node(null, null);
        this.tail = new Node(this.head, null);
        this.head.next = this.tail;

        this.sortAttr = sortAttr;
    }

    isEmpty() {
        return this.head.next == this.tail;
    }

    add(node) {
        var prevNode = this.findLastNodeBefore(node);
        this.addAfter(prevNode, node);
    }

    findLastNodeBefore(node) {
        if (this.isEmpty()) {
            return this.head;
        }
        var val = node.item[this.sortAttr];
        var cursor = this.head.next;
        while (cursor != this.tail) {
            var curVal = cursor.item[this.sortAttr];
            if (curVal > val) {
                break;
            }
            cursor = cursor.next;
        }

        return cursor.prev;
    }

    addAfter(node, nodeToAdd) {
        // update forward link;
        nodeToAdd.next = node.next;
        node.next = nodeToAdd;

        // update backward link;
        nodeToAdd.prev = node;
        nodeToAdd.next.prev = nodeToAdd;
    }

    remove(node) {
        if (node == this.head || node == this.tail) {
            // head and tail cannot be removed;
            return;
        }
        console.log(node);

        node.prev.next = node.next;
        node.next.prev = node.prev;

        node.next = null;
        node.prev = null;

        console.log(this.head);
    }

    getItems() {
        var items = [];
        var cursor = this.head.next;
        while (cursor != this.tail) {
            items.push(cursor.item);
            cursor = cursor.next;
        }
        return items;
    }
}
