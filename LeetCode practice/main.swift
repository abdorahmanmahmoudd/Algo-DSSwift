//
//  main.swift
//  LeetCode practice
//
//  Created by Abdelrahman Ali on 03/10/2020.
//  Copyright © 2020 Abdo. All rights reserved.
//

import Foundation

// #1 1. Two Sum
//func twoSum(_ nums: [Int], _ target: Int) -> [Int] {
//
//    for i in 0..<nums.count {
//
//        let sub = target - nums[i]
//        for j in 0..<nums.count {
//
//            if j != i {
//                if nums[j] == sub {
//                    return [i,j]
//                }
//            }
//        }
//    }
//    return []
//}

// MORE EFFECIENT SOLUTION O(1)

//func twoSum(_ nums: [Int], _ target: Int) -> [Int] {
//
//    var numsMap: [Int: Int] = [:]
//    for i in 0..<nums.count {
//        numsMap[nums[i]] = i
//    }
//
//    for i in 0..<nums.count {
//        let sub = target - nums[i]
//        if let subIndex = numsMap[sub], subIndex != i {
//            return [i, subIndex]
//        }
//    }
//    return []
//}

//let nums = [3,2,4]
//print(twoSum(nums, 6))

//------------------------------------------------------------------------------------------------

// #2 339. Nested List Weight Sum
//
// protocol NestedInteger {
//     // Return true if this NestedInteger holds a single integer, rather than a nested list.
//    func isInteger() -> Bool
//
//    // Return the single integer that this NestedInteger holds, if it holds a single integer
//    // The result is undefined if this NestedInteger holds a nested list
//    func getInteger() -> Int
//
//    // Set this NestedInteger to hold a single integer.
//    mutating func setInteger(value: Int)
//
//    // Set this NestedInteger to hold a nested list and adds a nested integer to it.
//    mutating func add(elem: NestedInteger)
//
//    // Return the nested list that this NestedInteger holds, if it holds a nested list
//    // The result is undefined if this NestedInteger holds a single integer
//    func getList() -> [NestedInteger]
//}
//
//struct NLItem: NestedInteger {
//
//    var value: Int?
//    var list: [NestedInteger]?
//
//    func isInteger() -> Bool {
//        return value == nil ? false : true
//    }
//
//    func getInteger() -> Int {
//        return value!
//    }
//
//    mutating func setInteger(value: Int) {
//        list = nil
//        self.value = value
//    }
//
//    mutating func add(elem: NestedInteger) {
//        if list == nil {
//            list = []
//        }
//        list!.append(elem)
//        value = nil
//    }
//
//    func getList() -> [NestedInteger] {
//        return list!
//    }
//}
//
//func depthSum(_ nestedList: [NestedInteger]) -> Int {
//    return depthSum(nestedList, 1)
//}
//
//func depthSum(_ nestedList: [NestedInteger], _ depthLevel: Int) -> Int {
//    var sum = 0
//    for item in nestedList {
//        if item.isInteger() {
//            sum += item.getInteger() * depthLevel
//        } else {
//            sum += depthSum(item.getList(), depthLevel + 1)
//        }
//    }
//    return sum
//}
//
//let one = NLItem(value: 1, list: nil)
//let two = NLItem(value: 2, list: nil)
//let three = NLItem(value: 3, list: nil)
//let list1 = NLItem(value: nil, list: [one, one])
//let list2 = NLItem(value: nil, list: [three])
//let list3 = NLItem(value: nil, list: [two, list2])
//let list4 = NLItem(value: nil, list: [one, list3])
//
//let testList = [list1, two, list4]
//print(testList)
//print(depthSum(testList))

//------------------------------------------------------------------------------------------------

// #3 88. FB: Merge Sorted Array

//func merge(_ nums1: inout [Int], _ m: Int, _ nums2: [Int], _ n: Int) {
//
//    var result = Array<Int>()
//
//    var i = 0
//    var j = 0
//
//    while i < m, j < n {
//
//        if nums1[i] < nums2[j] {
//            result.append(nums1[i])
//            i += 1
//
//        } else if nums1[i] > nums2[j] {
//            result.append(nums2[j])
//            j += 1
//
//        } else {
//            result.append(nums2[j])
//            result.append(nums1[i])
//            j += 1
//            i += 1
//        }
//    }
//
//    if i < m {
//        result.append(contentsOf: nums1[i..<m])
//    }
//
//    if j < n {
//        result.append(contentsOf: nums2[j..<n])
//    }
//
//    nums1 = result
//}
//
//var n = [2,0]
//var m = [1]
//merge(&n, 1, m, 1)
//print(n)

//------------------------------------------------------------------------------------------------


// FB: 23. Merge k Sorted Lists -> *HARD*

//1- use devide and conquere approach to pair every 2 lists together and sort them
//2- Pair every 2 lists together
//3- sort them and return the sorted list to be merge with similar one
//
//L1  L2  L3  L4
// \  /    \  /
//  L5      L6
//    \.    /
//       L7
//
//O(Log K) for pairing * O(N) for sorting -> O(N log K)
//O(1)

//
//public class ListNode {
//    public var val: Int
//    public var next: ListNode?
//    public init() { self.val = 0; self.next = nil; }
//    public init(_ val: Int) { self.val = val; self.next = nil; }
//    public init(_ val: Int, _ next: ListNode?) { self.val = val; self.next = next; }
//}
//
//extension ListNode: CustomStringConvertible {
//    public var description: String {
//        var result = "\(val) -> "
//        var tempNode = self.next
//        while tempNode != nil {
//            result.append("\(tempNode!.val) ->")
//            tempNode = tempNode?.next
//        }
//        return result
//    }
//}
//
//func mergeKLists(_ lists: [ListNode?]) -> ListNode? {
//
//    // pair lists
//    let listsCount = lists.count
//    var listsHead = lists
//    var interval = 1
//
//    while interval < listsCount {
//
//        for i in stride(from: 0, to: listsCount - interval, by: interval * 2) {
//            // merge every 2 lists together into the first one
//            listsHead[i] = merge2Lists(listsHead[i], listsHead[i+interval])
//        }
//        // Modify the interval to point to the correct lists
//        interval *= 2
//    }
//    return listsHead.first ?? nil
//}
//
//private func merge2Lists(_ list1: ListNode?, _ list2: ListNode?) -> ListNode? {
//
//    // Create a pointer to track the nodes in sorted order
//    var point: ListNode? = ListNode(0)
//    // A head to keep the head of the point
//    let head = point!
//
//    // temp variables
//    var tempL1 = list1
//    var tempL2 = list2
//
//    // Loop through the 2 lists unless one of them is done
//    while tempL1 != nil, tempL2 != nil {
//
//        if tempL1!.val <= tempL2!.val {
//
//            point!.next = tempL1
//            tempL1 = tempL1!.next
//
//        } else {
//
//            point!.next = tempL2
//            tempL2 = tempL1
//            tempL1 = point!.next?.next
//        }
//        point = point!.next
//    }
//
//
//    // Check if some nodes not sorted yet
//    if let l1 = tempL1 {
//        point!.next = l1
//    } else if let l2 = tempL2 {
//        point!.next = l2
//    }
//
//    // Return head.next which is point second node
//    return head.next
//}
//
//let node1: ListNode = ListNode(1, ListNode(2))
//let node2: ListNode = ListNode(-1, ListNode(10))
//let list: [ListNode] = [node1, node2]
//print(mergeKLists(list) ?? "")

//------------------------------------------------------------------------------------------------

/*

 FB: Product of Array Except Self
1- for loop on: nums to calculate the product of the privouse item for each num
2- for loop (i = answerLength, i>=0, i--) with a variable holding the multiplication result for each item and then calculate the final result by multiplay it with last one.

nums: [2,3,4]
answer: [1,2,6]
answer -> [12,8,6] // r 1, 4, 12
return anwer
*/

//func productExceptSelf(_ nums: [Int]) -> [Int] {
//
//  var answer = Array<Int>()
//  answer.append(1)
//
//  for i in 1..<nums.count {
//    answer.append(nums[i-1] * answer[i-1])
//  }
//
//  var r = 1
//  for i in stride(from: answer.count - 1, to: -1, by: -1) {
//    answer[i] = answer[i] * r
//    r = r * nums[i]
//  }
//
//  return answer
//}

//------------------------------------------------------------------------------------------------

/*
FB: 29. Divide Two Integers
1- Double the devisor until we find the highest value <= divindend -> highest power of two - highest double

2- finding which power of two fits in the dividend, each time we find one we half the power of two and the highest double by using the right bit shit operator

3- quotient = the sum of power of two's we found

dividend = 93706
divisor = 157

hd = 160768
hpot = 2^10

quotient = 0

while(divisor <= dividend) { //157 <= 93703 \\ 157 <= 13309 // 157 <= 3261

if dividend >= hd { // 93703 >= 80394 // 13309 >= 40192 //  ... //  13309 >= 10048

quotient += hpot // 2^9 // 2^7
dividend -= hd // 93703 - 80394 // 13309 - 10048
}

hd >>= 1 // 160768 / 2 = 80384 \\ 40192 ...  // 10048
hpot >>= 1 // 2^9 \\ 2^8 // 2^7
}

return  quotient

*/
//func divide(_ dividend: Int, _ divisor: Int) -> Int {
//
//    var dividendVar = dividend
//    var highestDouble = divisor
//    var highestPowerOfTwo = 1
//    while (highestDouble + highestDouble <= dividend) {
//        highestDouble += highestDouble
//        highestPowerOfTwo += highestPowerOfTwo
//    }
//
//    var quotient = 0
//
//    while (divisor <= dividendVar) {
//
//        if (dividendVar >= highestDouble) {
//            quotient += highestPowerOfTwo
//            dividendVar -= highestDouble
//        }
//
//        highestDouble = highestDouble >> 1
//        highestPowerOfTwo = highestPowerOfTwo >> 1
//    }
//
//    return quotient
//}

// To deal with negatives/positives -> convert everything to negatives since it covers bigger range (-2^31) -> (2^31 - 1)
//func divide(_ dividend: Int, _ divisor: Int) -> Int {
//    // handle special case of 2^31, overflow.
//    if dividend == Int32.min && divisor == -1 {
//        return Int(Int32.max)
//    }
//
//    // convert both to negatives and count negatives for later
//    var dividendVar = dividend
//    var divisorVar = divisor
//    var negatives = 2
//    if dividendVar > 0 {
//        negatives -= 1
//        dividendVar = -dividendVar
//    }
//    if divisorVar > 0 {
//        negatives -= 1
//        divisorVar = -divisorVar
//    }
//
//
//    var highestDouble = divisorVar
//    var highestPowerOfTwo = -1
//
//    while (highestDouble + highestDouble >= dividendVar) {
//        highestDouble += highestDouble
//        highestPowerOfTwo += highestPowerOfTwo
//    }
//
//    var quotient = 0
//    while divisorVar >= dividendVar {
//
//        if dividendVar <= highestDouble {
//            quotient += highestPowerOfTwo
//            dividendVar -= highestDouble
//        }
//
//        highestDouble = highestDouble >> 1
//        highestPowerOfTwo = highestPowerOfTwo >> 1
//    }
//
//    // if one of them was negative then the result should be negative
//    if negatives != 1 {
//        return -quotient
//    }
//
//    return quotient
//}
//
//print(divide(-2147483648,-1))


//------------------------------------------------------------------------------------------------

////FB: Move Zeroes
//
//func moveZeroes(_ nums: inout [Int]) {
//
//    var lastNonZeroItem = 0
//
//    for i in 0..<nums.count {
//
//        if nums[i] != 0 {
//            nums.swapAt(i, lastNonZeroItem)
//            lastNonZeroItem += 1
//        }
//    }
//}


//------------------------------------------------------------------------------------------------

//FB: 523. Continuous Subarray Sum
/*

1- sum each num with the prevous sums starting from the previous num -> the possible continous sums
2- iterate through the sums and check sum % k == 0

O(N^2) time
O(N) space
*/
//func checkSubarraySum(_ nums: [Int], _ k: Int) -> Bool {
//
//  guard nums.count > 0 else {
//    return false
//  }
//
//  var sums = Array<Int>()
//  sums.append(nums[0])
//
//  for i in 1..<nums.count {
//    sums.append(nums[i] + sums.last!)
//  }
//
//  for start in 0..<nums.count {
//    for end in start+1..<nums.count {
//      let sum = sums[end] - sums[start] + nums[start]
//      if sum == k || ( k != 0 && sum % k == 0) {
//        return true
//      }
//    }
//  }
//  return false
//}

// More Optimized solution using hashMap O(N) time and space O(min(n,k))

/*

1- a hashmap -> for each num get the sum and then we store the sum%k in it with the num index
2- if we encountered same reminder with different index that means there is such a number equals to (reminder+n*k)  where n > 0, passed where sum of them % k == 0 -> return true

O(N) time
O(min(k,m)) space

*/

//func checkSubarraySum(_ nums: [Int], _ k: Int) -> Bool {
//  var map: [Int: Int] = [:]
//  var sum = 0
//  map[0] = -1
//  for i in 0..<nums.count {
//    sum += nums[i]
//    if k != 0 {
//      sum = sum%k
//    }
//    if map[sum] != nil {
//      let index = map[sum]
//      if i - index! > 1 {
//        return true
//      }
//    } else {
//      map[sum] = i
//    }
//  }
//  return false
//}
//
//print(checkSubarraySum([0,0], 0))

//------------------------------------------------------------------------------------------------

// FB: 560. Subarray Sum Equals K

/*

-> sum[i] - sum[j] = k
 The idea behind this approach is as follows: If the cumulative sum(repreesnted by sum[i]sum[i] for sum upto i^{th}i
 th index) upto two indices is the same, the sum of the elements lying in between those indices is zero. Extending the same thought further, if the cumulative sum upto two indices,
 say i and j is at a difference of kk i.e. if sum[i] - sum[j] = k, sum[i]−sum[j]=k, the sum of elements lying between indices ii and jj is kk.

1- a hashmap to keep track of the cumulative sums for each index
2- if encountered a repeated cumulative sum, that means the elements between them equals to K. answer += 1


O(N) time
O(min(k,m)) space

*/
//func subarraySum(_ nums: [Int], _ k: Int) -> Int {
//  var map: [Int: Int] = [:]
//  var sum = 0
//  var count = 0
//  map[0] = 1
//
//  for i in 0..<nums.count {
//    sum += nums[i]
//
//    if map[sum-k] != nil {
//      count += map[sum-k]!
//    }
//    map[sum] = (map[sum] == nil ? 1 : map[sum]! + 1 )
//  }
//  return count
//}
//
//print(subarraySum([1,1,1], 2))

//------------------------------------------------------------------------------------------------

//FB: 215. Kth Largest Element in an Array

//func findKthLargest(_ nums: [Int], _ k: Int) -> Int {
//  var kVar = k
//  let sorted = nums.sorted()
//  for i in stride(from: sorted.count - 1, to: -1, by: -1) {
//    if kVar == 1 {
//      return sorted[i]
//    }
//    kVar -= 1
//  }
//  return 0
//}
//
//
//print(findKthLargest([3,2,1,5,6,4], 2))

//------------------------------------------------------------------------------------------------

//FB: 71. Simplify Path
/*
 1- separate the strings by `/` since every sub path should be separeted by /.
 2- process the separated array strings by pushing it into a stack
 3- join it back with `/`
 
 O(N) time compx
 O(N) space

 */

//
//func simplifyPath(_ path: String) -> String {
//    var cp = "/"
//    let pathArr = Array(path.split(separator: "/"))
//    var pathStck = [String]()
//    for path in pathArr {
//        if path == ".." {
//            _ = pathStck.popLast()
//            continue
//        } else if path == "." {
//            continue
//        }
//        pathStck.append(String(path))
//    }
//    cp += pathStck.joined(separator: "/")
//    if cp.last == "/", cp.count > 1 {
//        cp.removeLast()
//    }
//    return cp
//}



//print(simplifyPath("/home/foo/.ssh/../.ssh2/authorized_keys/")) // -> /home/foo/.ssh2/authorized_keys
//print(simplifyPath("/a//./")) // "/a"
//print(simplifyPath("/.../")) // "/"
//print(simplifyPath("/./")) // "/"
//print(simplifyPath("/a/./b/../../c/")) // /c
//print(simplifyPath("/a/../../b/../c//.//")) // /a/b/c


//------------------------------------------------------------------------------------------------

/*

FB: 270. Closest Binary Search Tree Value

 -> approach 1: deleted
// bfs through the tree nodes
// for each one we calculate node.val - target to get the smallest value which indicates the closest node.
 time complexity O(N)
 space O(N)
 
 -> approach 2: using binary search instead of bfs
time complexity O(H) : H is the tree depth
space O(1)

*/
public class TreeNode {
    public var val: Int
    public var left: TreeNode?
    public var right: TreeNode?
    public init() { self.val = 0; self.left = nil; self.right = nil; }
    public init(_ val: Int) { self.val = val; self.left = nil; self.right = nil; }
    public init(_ val: Int, _ left: TreeNode?, _ right: TreeNode?) {
        self.val = val
        self.left = left
        self.right = right
    }
}

extension TreeNode: CustomStringConvertible {
    public var description: String {
        diagram(for: self)
    }
    
    private func diagram(for node: TreeNode?,
                         _ top: String = "",
                         _ root: String = "",
                         _ bottom: String = "") -> String {
        guard let node = node else {
            return root + "nil\n"
        }
        if node.left == nil && node.right == nil {
            return root + "\(node.val)\n"
        }
        return diagram(for: node.right,
                       top + " ", top + "┌──", top + "│ ")
            + root + "\(node.val)\n"
            + diagram(for: node.left,
                      bottom + "│ ", bottom + "└──", bottom + " ")
    }
}
//
//func closestValue(_ root: TreeNode?, _ target: Double) -> Int {
//    var gap = Double.infinity
//    var head = root
//    var closest = root
//    while let current = head {
//        if abs(Double(current.val) - target) < gap {
//            gap = abs(Double(current.val) - target)
//            closest = current
//        }
//        head = target < Double(current.val) ? current.left : current.right;
//    }
//    return closest!.val
//}
//
//print(closestValue(nil, 123))

//------------------------------------------------------------------------------------------------

/*
 FB: 621. Task Scheduler
1- Sort the tasks based on the frequency
2- calculate the idle time by = (f_max-1) * n
3- iterate through the rest of the tasks and for each one decrease the idle time by min(f_max-1, f_task), where idle time >= 0
4- return idleTime + tasks number

[A, B, B], n = 2

[a:1, b:2] -> sorted [b:2, a:1]
idleTime = (2-1)*2 = 2
for taskType in tasks {
  idleTime -= min(f_max, taskType.freq)
}
if idleTime < 0 {
idleTime = 0
}
return tasks.count + idleTime

O(N Total number of different tasks(26)) time complexity
O(1) space comp
*/
//
//func leastInterval(_ tasks: [Character], _ n: Int) -> Int {
//    var freqs = Array(repeating: 0, count: 26)
//    for task in tasks {
//        let taskIndex = Int(task.asciiValue! - 65)
//        freqs[taskIndex] += 1
//    }
//    freqs = freqs.sorted(by: >)
//    let maxFreq = freqs.first!
//    var idu = (maxFreq-1) * n
//    for i in 1..<freqs.count {
//        idu -= min(maxFreq - 1, freqs[i])
//    }
//    idu = idu < 0 ? 0 : idu
//    return tasks.count + idu
//}
//
//print(leastInterval(["A","A","A","B","B","B"], 2))


//------------------------------------------------------------------------------------------------

/*
 FB: 364. Nested List Weight Sum II

 1- get the depth of the tree
 2- recursionly calculate the sums with decreasing the depth level by 1 for each level
 
 O(N) time complexity
 O(N) space complexity
 
*/

//func depthSumInverse(_ nestedList: [NestedInteger]) -> Int {
//    var depth = 0
//    getDepth(nestedList, &depth)
//    print(depth)
//    return depthSum(nestedList, depth)
//}
//
//private func getDepth(_ nestedList: [NestedInteger], _ depth: inout Int) {
//    var que: Array<NestedInteger> = []
//    var il = 0
//    for item in nestedList {
//        que.append(item)
//    }
//    while !que.isEmpty {
//        il = que.count
//        depth += 1
//        while il > 0 {
//            let item = que.removeFirst()
//            if !item.isInteger() {
//                que.append(contentsOf: item.getList())
//            }
//            il -= 1
//        }
//    }
//}
//
//private func depthSum(_ nestedList: [NestedInteger], _ depthLevel: Int) -> Int {
//    var sum = 0
//    print(depthLevel)
//    for item in nestedList {
//        if item.isInteger() {
//            sum += item.getInteger() * depthLevel
//        } else {
//            sum += depthSum(item.getList(), depthLevel - 1)
//        }
//    }
//    return sum
//}

//------------------------------------------------------------------------------------------------


/*
FB:  938. Range Sum of BST

1- traverse all nodes and sum the values if node.val >= L && node.val <= R

O(N) time
O(1) space
*/
//func rangeSumBST(_ root: TreeNode?, _ L: Int, _ R: Int) -> Int {
//    guard let root = root else {
//        return 0
//    }
//    var que: Array<TreeNode?> = []
//    que.append(root)
//    var sum = 0
//    while let current = que.popLast(), current != nil {
//        if current!.val >= L, current!.val <= R {
//            sum += current!.val
//        }
//        if current!.val <= R, let right = current?.right {
//            que.append(right)
//        }
//        if current!.val >= L, let left = current?.left {
//            que.append(left)
//        }
//    }
//    return sum
//}

//------------------------------------------------------------------------------------------------

/*
FB:  121. Best Time to Buy and Sell Stock
 
 1- looking for the smallest number that have the biggest (differece) peek after it.
 O(N) time
 O(1) space
*/

//func maxProfit(_ prices: [Int]) -> Int {
//    var minPrice = Int.max
//    var maxProfit = 0
//    for price in prices {
//        if price <= minPrice { // find the smallest number (valley)
//            minPrice = price
//        } else if (price - minPrice >= maxProfit) { // after that looking for the biggest peek (max profit) from that number
//            maxProfit = price - minPrice
//        }
//    }
//    return maxProfit
//}


//------------------------------------------------------------------------------------------------

/*
for i in 0..<indices.count -> pick the string[i] and insert it in indices[i] place

[cab], indices = [3,1,2]
*/

//func restoreString(_ s: String, _ indices: [Int]) -> String {
//    var r = Array(s)
//    let copyS = Array(s)
//    for i in 0..<indices.count {
//        r[indices[i]] = Character(extendedGraphemeClusterLiteral: copyS[i])
//    }
//    return String(r)
//}
//
//print(restoreString("codeleet", [4,5,6,7,0,2,1,3]))

//------------------------------------------------------------------------------------------------

// FB: 1266. Minimum Time Visiting All Points

//func minTimeToVisitAllPoints(_ points: [[Int]]) -> Int {
//    guard points.count > 0 else {
//        return 0
//    }
//    var ans = 0
//    var c = points.first!
//    for p in points {
//        let xd = abs(p[0] - c[0])
//        let yd = abs(p[1] - c[1])
//        if xd >= yd {
//            ans += xd
//        } else {
//            ans += yd
//        }
//        c = p
//    }
//    return ans
//}

//------------------------------------------------------------------------------------------------

/*
FB: 973. K Closest Points to Origin
 
1- calculate the Euclidean distance for every point root(pow((x1 - x2), 2) + pow(y1 - y2), 2))
2- sort them
3- refurn first K elements

O(NlogN) time, bcs of the sorting
O(N) space
 */

// Using Swift Sorting func -> 1068 ms
//func kClosest(_ points: [[Int]], _ K: Int) -> [[Int]] {
//    var d = Array<(distance: Double, point: [Int])>()
//    for i in 0..<points.count {
//        let x2 = pow(Double(points[i][0]), 2)
//        let y2 = pow(Double(points[i][1]), 2)
//        let ed =  (x2 + y2).squareRoot()
//        d.append((ed, points[i]))
//    }
//    d.sort { (point1, point2) -> Bool in
//        return point1.distance > point2.distance
//    }
//    return d.suffix(K).map({ $0.point })
//}

// Using Merge sort -> 1220 ms
//func kClosest(_ points: [[Int]], _ K: Int) -> [[Int]] {
//    var d = Array<(distance: Double, point: [Int])>()
//    d.reserveCapacity(points.count)
//    for i in 0..<points.count {
//        let x2 = pow(Double(points[i][0]), 2)
//        let y2 = pow(Double(points[i][1]), 2)
//        let ed =  (x2 + y2).squareRoot()
//        d.append((ed, points[i]))
//    }
//    d = mergeSort(arr: d)
//    return d.suffix(K).map({ $0.point })
//}
//
//private func mergeSort(arr: [(Double,[Int])]) -> [(Double,[Int])] {
//    guard arr.count > 1 else {
//        return arr
//    }
//    let middle = arr.count / 2
//    let left = mergeSort(arr: Array(arr[..<middle]))
//    let right = mergeSort(arr: Array(arr[middle...]))
//    return mergeSort(left, right)
//}
//private func mergeSort(_ left: [(Double, [Int])], _ right: [(Double, [Int])]) -> [(Double, [Int])] {
//    var leftIndex = 0
//    var rightIndex = 0
//    var result = [(Double, [Int])]()
//    while leftIndex < left.count && rightIndex < right.count {
//        if left[leftIndex].0 > right[rightIndex].0 {
//            result.append(left[leftIndex])
//            leftIndex += 1
//        } else if left[leftIndex].0 < right[rightIndex].0 {
//            result.append(right[rightIndex])
//            rightIndex += 1
//        } else {
//            result.append(right[rightIndex])
//            rightIndex += 1
//            result.append(left[leftIndex])
//            leftIndex += 1
//        }
//    }
//    if leftIndex < left.count {
//        result.append(contentsOf: left[leftIndex...])
//    }
//    if rightIndex < right.count {
//        result.append(contentsOf: right[rightIndex...])
//    }
//    return result
//}

//print(kClosest([[68,97],[34,-84],[60,100],[2,31],[-27,-38],[-73,-74],[-55,-39],[62,91],[62,92],[-57,-67]], 5))

//------------------------------------------------------------------------------------------------

/*
 FB: 1213. Intersection of Three Sorted Arrays

1- Create a hashmap of the number is the key and the counter is the value
2- return the number when counter reach to 3

time O(N)
space O(N)

*/
//func arraysIntersection(_ arr1: [Int], _ arr2: [Int], _ arr3: [Int]) -> [Int] {
//  var countMap = [Int: Int]()
//  var ans = [Int]()
//  for i in 0..<arr1.count {
//    if countMap[arr1[i]] == nil {
//      countMap[arr1[i]] = 1
//    } else {
//      countMap[arr1[i]]! += 1
//    }
//  }
//  for i in 0..<arr2.count {
//    if countMap[arr2[i]] == nil {
//      countMap[arr2[i]] = 1
//    } else {
//      countMap[arr2[i]]! += 1
//    }
//  }
//  for i in 0..<arr3.count {
//    if countMap[arr3[i]] == nil {
//      countMap[arr3[i]] = 1
//    } else {
//      countMap[arr3[i]]! += 1
//    }
//    if countMap[arr3[i]] == 3 {
//      ans.append(arr3[i])
//    }
//  }
//  return ans
//}
//
//let arr1 = [6,16,23,37,45,54,58,60,66,87,95,102,135,136,145,146,159,161,170,171,175,178,200,208,209,211,215,217,218,227,229,238,239,276,289,295,298,313,318,324,331,333,340,344,355,357,372,373,374,376,379,390,394,395,399,413,418,419,425,431,432,436,449,458,481,484,487,489,494,501,511,515,518,524,526,528,529,534,542,544,547,552,559,564,565,571,581,589,590,595,607,618,620,641,652,663,664,669,672,680,686,694,702,713,715,729,735,746,755,769,773,774,778,780,791,793,802,804,808,810,812,816,822,827,831,841,842,850,851,861,865,877,883,891,904,907,910,912,913,915,917,934,945,958,960,971,974,976,997,999,1008,1010,1011,1015,1027,1037,1040,1045,1055,1056,1070,1090,1099,1114,1118,1122,1125,1132,1133,1141,1143,1146,1153,1159,1165,1168,1170,1172,1173,1179,1181,1184,1207,1214,1218,1219,1239,1247,1255,1267,1273,1282,1285,1295,1300,1304,1312,1326,1346,1358,1360,1362,1367,1375,1396,1397,1402,1410,1412,1416,1418,1420,1424,1425,1435,1443,1447,1464,1470,1479,1491,1502,1507,1509,1515,1520,1531,1537,1539,1556,1562,1563,1565,1577,1582,1583,1587,1589,1619,1642,1645,1648,1652,1662,1665,1677,1678,1695,1707,1711,1713,1725,1727,1731,1736,1744,1747,1751,1757,1771,1776,1783,1784,1787,1797,1802,1809,1812,1823,1827,1828,1829,1833,1836,1847,1854,1860,1867,1873,1874,1880,1887,1888,1897,1911,1913,1919,1923,1931,1948,1951,1954,1964,1965,1967,1969,1971,1973,1982,1988]
//let arr2 = [21,33,38,50,53,57,64,78,81,82,89,96,97,117,123,131,140,147,149,152,160,161,173,178,185,186,200,233,234,236,245,250,256,288,294,314,318,323,327,330,337,338,347,350,352,369,384,385,386,391,395,396,397,407,410,425,435,449,458,461,469,472,476,488,489,490,506,512,522,533,537,545,560,561,562,564,572,588,596,601,603,617,651,653,656,659,661,673,678,684,685,698,699,701,712,716,719,725,726,727,729,732,743,744,747,753,759,771,772,773,780,783,786,799,827,830,834,836,837,840,843,847,850,853,860,866,870,879,883,888,893,912,914,924,929,931,938,946,948,951,959,972,980,981,985,993,996,1010,1011,1014,1015,1022,1025,1029,1044,1048,1050,1053,1057,1066,1067,1070,1080,1083,1093,1095,1100,1102,1137,1151,1152,1155,1159,1170,1191,1192,1195,1211,1214,1222,1228,1229,1232,1247,1249,1256,1275,1276,1279,1280,1281,1292,1293,1306,1324,1326,1332,1348,1362,1363,1368,1386,1397,1401,1407,1408,1411,1417,1419,1421,1424,1430,1433,1443,1445,1457,1467,1471,1472,1484,1486,1488,1498,1504,1505,1521,1526,1540,1549,1550,1555,1558,1559,1563,1565,1578,1582,1584,1600,1601,1603,1612,1623,1626,1635,1640,1644,1652,1653,1654,1655,1658,1661,1669,1670,1703,1714,1726,1734,1739,1747,1749,1759,1760,1770,1796,1815,1821,1826,1838,1840,1841,1850,1853,1855,1857,1858,1859,1878,1882,1886,1888,1892,1896,1897,1899,1909,1911,1918,1920,1922,1937,1943,1953,1962,1963,1964,1980,1993,1995]
//let arr3 =
//[4,7,8,9,12,21,25,29,32,37,39,48,55,63,65,71,72,81,82,83,96,97,104,109,114,116,118,120,122,124,127,131,136,154,161,165,166,177,182,184,187,200,203,213,223,226,230,240,278,283,286,309,313,315,337,338,349,354,357,362,363,364,366,369,377,380,381,384,393,399,409,410,416,422,435,441,444,452,459,460,462,463,464,467,470,471,485,491,511,515,536,553,557,571,573,576,577,594,598,599,601,618,619,635,642,647,652,661,671,674,680,697,705,712,713,730,733,735,746,754,759,767,768,777,781,787,801,804,808,814,819,831,835,847,859,860,861,872,888,890,892,899,902,907,916,926,928,929,932,937,942,949,960,968,970,980,986,993,996,1005,1006,1007,1009,1014,1017,1026,1028,1031,1036,1041,1043,1047,1048,1054,1062,1066,1069,1072,1075,1079,1089,1090,1091,1094,1105,1111,1112,1113,1122,1139,1142,1143,1148,1157,1159,1160,1162,1163,1187,1190,1202,1219,1235,1244,1247,1249,1250,1261,1265,1279,1296,1297,1308,1309,1313,1315,1320,1323,1340,1344,1358,1370,1372,1375,1380,1415,1418,1419,1422,1432,1438,1450,1464,1466,1471,1473,1476,1479,1490,1503,1508,1511,1521,1535,1538,1541,1562,1571,1572,1576,1583,1602,1618,1620,1626,1628,1630,1647,1650,1662,1664,1665,1667,1669,1679,1686,1687,1705,1707,1742,1745,1750,1757,1784,1793,1813,1825,1826,1827,1846,1854,1863,1871,1872,1878,1886,1888,1898,1909,1913,1915,1916,1919,1932,1935,1939,1948,1970,1984,1996]
//print(arraysIntersection(arr1, arr2, arr3))

//------------------------------------------------------------------------------------------------

/*
 
 FB: 496. Next Greater Element I

// Brute force
1- for each element in nums1 find it in nums2
2- loop in nums2 from num1 index till the end to find the first greater element.
if it's not exists append -1
time O(N^2)
space O(1)
 */

// Brute force -> can be enhanced using a hashmap to store nums with its indices.
//func nextGreaterElement(_ nums1: [Int], _ nums2: [Int]) -> [Int] {
//  var ans = [Int]()
//  for i in 0..<nums1.count {
//    for j in 0..<nums2.count {
//      var nge = -1
//      if nums1[i] == nums2[j] {
//        for x in j..<nums2.count {
//          if nums2[x] > nums2[j] {
//            nge = nums2[x]
//            break
//          }
//        }
//        ans.append(nge)
//        break
//      }
//    }
//  }
//  return ans
//}

/*
 // Stack + hashmap approach
 1- construct evey possible pair of (num, greateNextElement) into the map using the stack
 2- iterate over the nums only once to look into the map for results
 time O(N+M)
 space O(N+M)
 */

// Stack + hashmap approach
//func nextGreaterElement(_ nums1: [Int], _ nums2: [Int]) -> [Int] {
//  var stack = Array<Int>()
//  var map = [Int: Int]()
//  var ans = [Int]()
//  for i in 0..<nums2.count {
//    while !stack.isEmpty, nums2[i] > stack.last! {
//      map[stack.removeLast()] = nums2[i]
//    }
//    stack.append(nums2[i])
//  }
//  while !stack.isEmpty {
//    map[stack.removeLast()] = -1
//  }
//  for i in 0..<nums1.count {
//    if let num = map[nums1[i]] {
//        ans.append(num)
//    }
//  }
//  return ans
//}
//
//print(nextGreaterElement([4,1,2], [1,3,4,2])) // [-1,3,-1]


//------------------------------------------------------------------------------------------------

/*
 FB: 503. Next Greater Element II
 note: similar to the previous question but its only 1 circular array

 */

// My brute force -> 548 ms
//func nextGreaterElements(_ nums: [Int]) -> [Int] {
//    var ans = [Int]()
//    for j in 0..<nums.count {
//        var nge = -1
//        var i = j+1
//        var found = false
//        while j != i, i < nums.count {
//            if nums[i] > nums[j] {
//                nge = nums[i]
//                found = true
//                break
//            }
//            i += 1
//        }
//        if !found {
//            i = 0
//            while i < j {
//                if nums[i] > nums[j] {
//                    nge = nums[i]
//                    found = true
//                    break
//                }
//                i += 1
//            }
//        }
//        ans.append(nge)
//    }
//    return ans
//}

// Leetcode brute force -> 1072 ms
// func nextGreaterElements(_ nums: [Int]) -> [Int] {
//    var ans = [Int]()
//    for i in 0..<nums.count {
//        var nge = -1
//        for j in 1..<nums.count {
//            if nums[(i + j) % nums.count] > nums[i] {
//                nge = nums[(i + j) % nums.count]
//                break
//            }
//        }
//        ans.append(nge)
//    }
//    return ans
//}

/*
FB: 503. Next Greater Element II
1- traverse the nums array from the end and use a stack to push the nums with its index as the numbers may repeat
2- compare nums[i] to the stack.peek. if nums[i] < stack.peek, that means the stack.peek is a next greater element for nums[i]
3- else we pop all the smaller stack items until we found bigger one or the stack is empty then we push the nums[i] itself to act as a next greater element for the next nums

4- after finisht he first iteration, we iterate one more time because the array is circular and we still need to consider the nums to the left of the nums[i]

O(N) time
O(N) space

*/

//func nextGreaterElements(_ nums: [Int]) -> [Int] {
//    var ngeStack = Array<Int>()
//    var ans = Array(repeating: -1, count: nums.count)
//    for i in stride(from: 2 * nums.count - 1, to: -1, by: -1) {
//        let index = i % nums.count
//        while !ngeStack.isEmpty, nums[ngeStack.last!] <= nums[index] {
//            ngeStack.removeLast()
//        }
//        ans[index] = ngeStack.isEmpty ? -1 : nums[ngeStack.last!]
//        ngeStack.append(index)
//    }
//    return ans
//}
//
//
//print(nextGreaterElements([100,1,11,1,120,111,123,1,-1,-100])) //[120,11,120,120,123,123,-1,100,100,100]

//------------------------------------------------------------------------------------------------

/*
FB: 1249. Minimum Remove to Make Valid Parentheses

1- loop to count the '(' with ')' and if '(' is Zero while appending the candidate answer, do not append it.
2- check if '() counter  > 0 -> if so, that means there are some '(' came at the end of the string needs to be removed.
3- reverse the candidate and loop through it to remove '(' from last until '(' counter == 0
4- reverse the ans and return it
 
 O(N) time
O(N) space
*/

//func minRemoveToMakeValid(_ s: String) -> String {
//    var openCounter = 0
//    var candidate = ""
//    var ans = ""
//    for char in s {
//        if char == "(" {
//            openCounter += 1
//        } else if char == ")" {
//            if openCounter == 0 {
//                continue
//            }
//            openCounter -= 1
//        }
//        candidate.append(char)
//    }
//    if openCounter == 0 {
//        return candidate
//    }
//    for char in candidate.reversed() {
//        if char == "(", openCounter > 0 {
//            openCounter -= 1
//            continue
//        }
//        ans.append(char)
//    }
//    return String(ans.reversed())
//}
//
//print(minRemoveToMakeValid("())()((("))

//------------------------------------------------------------------------------------------------

/*
 FB: 173. Binary Search Tree Iterator
 
 1- construct an array by doing inORder traversal over the BST since its sorted.
 2- return the first element of the sorted array as the next()
 3- make an index to check if hasNext()
 
 O(N) time
 O(N) space
 
 */


//class BSTIterator {
//    var sortedArr: Array<Int>
//    var iteratorIndex: Int
//
//    init(_ root: TreeNode?) {
//        sortedArr = Array<Int>()
//        iteratorIndex = -1
//        inOrderTraversal(root)
//    }
//
//    private func inOrderTraversal(_ node: TreeNode?) {
//        if node == nil {
//            return
//        }
//        inOrderTraversal(node?.left)
//        sortedArr.append(node!.val)
//        inOrderTraversal(node?.right)
//    }
//
//    /** @return the next smallest number */
//    func next() -> Int {
//        iteratorIndex += 1
//        return sortedArr[iteratorIndex]
//    }
//
//    /** @return whether we have a next smallest number */
//    func hasNext() -> Bool {
//        return iteratorIndex + 1 < sortedArr.count
//    }
//}
//
//
//
//let n1 = TreeNode(3)
//let n5 = TreeNode(9)
//let n6 = TreeNode(20)
//let n3 = TreeNode(15, n5, n6)
//let root = TreeNode(7, n1, n3)
////let root: TreeNode? = nil
//let bstIterator = BSTIterator(root)
//print(bstIterator.next())
//print(bstIterator.next())
//print(bstIterator.hasNext())
//print(bstIterator.next())
//print(bstIterator.hasNext())
//print(bstIterator.next())
//print(bstIterator.hasNext())
//print(bstIterator.next())
//print(bstIterator.hasNext())

//------------------------------------------------------------------------------------------------

/*

FB: 438. Find All Anagrams in a String
1- create a hashmap to count all chars in p (key: char, value: count)
2- go through s by window of same size as p.count and for each window check chars count against the hashmap -> ans

time: O(Np + Ns)
space O(1) the hashmap will not contain more than 26 items
*/

//func findAnagrams(_ s: String, _ p: String) -> [Int] {
//    var pCount: [Character: Int] = [:]
//    var anagrams = [Int]()
//    for char in p {
//        pCount[char] = pCount[char] == nil ? 1 : pCount[char]! + 1
//    }
//    let sArr = Array(s)
//    let ps = p.count
//    var sCount: [Character: Int] = [:]
//    for (i, char) in sArr.enumerated() {
//        sCount[char] = sCount[char] == nil ? 1 : sCount[char]! + 1
//        if i >= ps {
//            let mostLeftCharIndex = i - ps
//            let mostLeftChar = sArr[mostLeftCharIndex]
//            if sCount[mostLeftChar] == 1 {
//                sCount[mostLeftChar] = nil
//            } else {
//                sCount[mostLeftChar]! -= 1
//            }
//        }
//        if pCount == sCount {
//            anagrams.append(i - ps + 1)
//        }
//    }
//    return anagrams
//}
//
//print(findAnagrams("abcdefghijkcbalmbacnopq", "abc"))

//------------------------------------------------------------------------------------------------

/*
242. Valid Anagram
time: O(Ns + Nt)
space: O(1)
*/

//func isAnagram(_ s: String, _ t: String) -> Bool {
//    guard s.count == t.count else {
//        return false
//    }
//    var sCount = [Character: Int]()
//    var tCount = [Character: Int]()
//    for char in s {
//        sCount[char] = sCount[char] == nil ? 0 : sCount[char]! + 1
//    }
//    for char in t {
//        tCount[char] = tCount[char] == nil ? 0 : tCount[char]! + 1
//    }
//    return tCount == sCount
//}
//
//print(isAnagram("rat", "car"))

//------------------------------------------------------------------------------------------------

/*
34. Find First and Last Position of Element in Sorted Array
 ** binary search **
time: O(LogN)
space: O(1)
*/


//func searchRange(_ nums: [Int], _ target: Int) -> [Int] {
//    let firstIndexOfTarget = binarySearch(nums, target, true)
//    if firstIndexOfTarget == nums.count || nums[firstIndexOfTarget] != target {
//        return [-1,-1]
//    }
//    let lastIndexOfTarget = binarySearch(nums, target) - 1
//    return [firstIndexOfTarget, lastIndexOfTarget]
//}
//
//private func binarySearch(_ nums: [Int], _ target: Int, _ first: Bool = false) -> Int {
//    var low = 0
//    var high = nums.count
//    while low < high {
//        let mid = low + (high - low) / 2
//        if target < nums[mid] || (first && nums[mid] == target) {
//            high = mid
//        } else {
//            low = mid + 1
//        }
//    }
//    return low
//}
//
//print(searchRange([1], 1))


//------------------------------------------------------------------------------------------------

/*
 FB: 278. First Bad Version
 ** binary search **
 */

//func firstBadVersion(_ n: Int) -> Int {
//    return binarySearch(1, n)
//}
//
//private func binarySearch(_ start: Int, _ end: Int) -> Int {
//    var low = start
//    var high = end
//    while low < high {
//        let mid = low + (high - low) / 2
//        if isBadVersion(mid) {
//            high = mid
//        } else {
//            low = mid + 1
//        }
//    }
//    return low
//}

//------------------------------------------------------------------------------------------------

/*
 567. Permutation in String
  
*/

//func checkInclusion(_ s1: String, _ s2: String) -> Bool {
//    var s1Count: [Character: Int] = [:]
//    for char in s1 {
//        s1Count[char] = s1Count[char] == nil ? 1 :  s1Count[char]! + 1
//    }
//    let s1Size = s1.count
//    let s2Arr = Array(s2)
//    var s2Count: [Character: Int] = [:]
//    for (i,char) in s2Arr.enumerated() {
//        s2Count[char] = s2Count[char] == nil ? 1 :  s2Count[char]! + 1
//        if i >= s1Size {
//            let mostLeftCharIndex = i - s1Size
//            let mostLeftChar = s2Arr[mostLeftCharIndex]
//            if s2Count[mostLeftChar] == 1 {
//                s2Count[mostLeftChar] = nil
//            } else {
//                s2Count[mostLeftChar]! -= 1
//            }
//        }
//        if s2Count == s1Count {
//            return true
//        }
//    }
//    return false
//}

//------------------------------------------------------------------------------------------------

/*
FB: 314. Binary Tree Vertical Order Traversal


1- A hashmap [vl: [TreeNode]] verticalLevel = 0
2- traverse the tree with in pre-order traversal + dfs
3- increase the vl when push left nodes and decrease it when we push right nodes.



    3
    /\
   /  \
   9   8
  /\  /\
 /  \/  \
 4  01   7

columnTable with BFS preorder traversal
-2: [4]
-1: [9]
0: [3, 0, 1]
1: [8]
2: [7]
 
 time: O(N)
 space: O(N)
*/

//func verticalOrder(_ root: TreeNode?) -> [[Int]] {
//    guard let root = root else {
//        return []
//    }
//    var columnsTable: [Int: [Int]] = [:] //  to keep track of the nodes corresponding to the vertical level (column)
//    var que = [(node: TreeNode, column: Int)]()
//    que.append((root, 0))
//    var minColumn = Int.max
//    /// BFS pre-order traversal -> BFS will gurantee the left-right order by default
//    while !que.isEmpty {
//        let pair = que.removeFirst()
//        if columnsTable[pair.column] == nil {
//            columnsTable[pair.column] = [pair.node.val]
//        } else {
//            columnsTable[pair.column]?.append(pair.node.val)
//        }
//        if let left = pair.node.left {
//            que.append((left, pair.column-1))
//        }
//        if let right = pair.node.right {
//            que.append((right, pair.column+1))
//        }
//        if minColumn > pair.column {
//            minColumn = pair.column
//        }
//    }
//    var ans = [[Int]](repeating: [Int](), count: columnsTable.count)
//    for level in columnsTable {
//        ans[level.key - minColumn] = level.value
//    }
//    return ans
//}

//------------------------------------------------------------------------------------------------

/*
 FB: 102. Binary Tree Level Order Traversal
*/

//func levelOrder(_ root: TreeNode?) -> [[Int]] {
//    guard let root = root else {
//        return []
//    }
//    var rowsTable: [Int: [Int]] = [:] //  to keep track of the nodes corresponding to the horizontal level (row)
//    var que = [TreeNode]()
//    que.append(root)
//    var nodesLeftInCurrentLevel = 0
//    var level = 0
//    /// BFS pre-order traversal -> BFS will gurantee the left-right order by default
//    while !que.isEmpty {
//        nodesLeftInCurrentLevel = que.count
//        while nodesLeftInCurrentLevel > 0 {
//            let node = que.removeFirst()
//            if rowsTable[level] == nil {
//                rowsTable[level] = [node.val]
//            } else {
//                rowsTable[level]?.append(node.val)
//            }
//            nodesLeftInCurrentLevel -= 1
//            if let left = node.left {
//                que.append(left)
//            }
//            if let right = node.right {
//                que.append(right)
//            }
//        }
//        level += 1
//    }
//    var ans = [[Int]](repeating: [Int](), count: rowsTable.count)
//    for level in rowsTable {
//        ans[level.key] = level.value
//    }
//    return ans
//}

//let n1 = TreeNode(3)
//let n5 = TreeNode(9)
//let n6 = TreeNode(20)
//let n3 = TreeNode(15, n5, n6)
//let root = TreeNode(7, n1, n3)
//print(levelOrder(root))


//------------------------------------------------------------------------------------------------

/*
133. Clone Graph

[[2,4],[1,3],[2,4],[1,3]]

n1 -> n2, n4
n2 -> n1, n3
 .
 .
 .

 
1- create a hashmap to track the cloned nodes
2- BFS through the nodes (graph) and create a clone for each node that are not cloned before
 3- if the node is not cloned before, we add to the que to clone it

time: O(N)
space: O(N)

*/

//public class Node {
//    public var val: Int
//    public var neighbors: [Node?]
//    public init(_ val: Int) {
//        self.val = val
//        self.neighbors = []
//    }
//}
//extension Node: CustomStringConvertible {
//    public var description: String {
//        return "\(val) -> \(neighbors.map({ "\($0!.val)" }).joined(separator: ", "))"
//    }
//}
//
//func cloneGraph(_ node: Node?) -> Node? {
//    guard let start = node else {
//        return nil
//    }
//    var que: [Node] = [start]
//    var clonesMap: [Int: Node] = [:]
//    while !que.isEmpty {
//        let or = que.removeLast()
//        var clone = Node(or.val)
//        if clonesMap[or.val] != nil {
//            clone = clonesMap[or.val]!
//        }
//        for node in or.neighbors {
//            if clonesMap[node!.val] == nil {
//                clonesMap[node!.val] = Node(node!.val)
//                que.append(node!)
//            }
//            clone.neighbors.append(clonesMap[node!.val]!)
//        }
//        clonesMap[clone.val] = clone
//    }
//    return clonesMap[start.val]
//}
//
//var n1 = Node(1)
//var n2 = Node(2)
//var n3 = Node(3)
//var n4 = Node(4)
//n1.neighbors.append(contentsOf: [n2,n4])
//n2.neighbors.append(contentsOf: [n1,n3])
//n3.neighbors.append(contentsOf: [n2,n4])
//n4.neighbors.append(contentsOf: [n1,n3])
//
//print(cloneGraph(n1))


//------------------------------------------------------------------------------------------------

/*
211. Design Add and Search Words Data Structure

- Implement a Trie data structure
- For searching functionality with '.' -> we have to check all chars after facing a dot.
 
 time: O(N.M) : M is the word length
 space: O(M)

*/

//class TrieNode<Element: Hashable> {
//    var value: Element?
//    var children: [Element: TrieNode]
//    var isTerminating = false
//    init(value: Element?, children: [Element: TrieNode] = [:]) {
//        self.value = value
//        self.children = children
//    }
//}
//
//
//class WordDictionary {
//
//    var root: TrieNode<Character>
//    /** Initialize your data structure here. */
//    init() {
//        root = TrieNode(value: nil)
//    }
//
//    /** Adds a word into the data structure. */
//    func addWord(_ word: String) {
//        var current = root
//        for char in word {
//            if current.children[char] == nil {
//                current.children[char] = TrieNode(value: char)
//            }
//            current = current.children[char]!
//        }
//        current.isTerminating = true
//    }
//
//    /** Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter. */
//    func search(_ word: String) -> Bool {
//       return searchInNode(word, node: root)
//    }
//
//    private func searchInNode(_ word: String, node: TrieNode<Character>) -> Bool {
//        var current = node
//        for (index, char) in zip(word.indices, word) {
//            if current.children[char] == nil {
//                if char == "." {
//                    for x in current.children {
//                        let subString = word[word.index(after: index)...]
//                        if searchInNode(String(subString), node: x.value) {
//                            return true
//                        }
//                    }
//                }
//                return false
//            } else {
//                current = current.children[char]!
//            }
//        }
//        return current.isTerminating
//    }
//}
//
//let wordDictionary = WordDictionary()
//wordDictionary.addWord("at")
//wordDictionary.addWord("and")
//wordDictionary.addWord("an")
//wordDictionary.addWord("add")
//print(wordDictionary.search("a")) // return False
//print(wordDictionary.search(".at")) // return false
//wordDictionary.addWord("bat")
//print(wordDictionary.search(".at"))
//print(wordDictionary.search("an."))
//print(wordDictionary.search("a.d."))
//print(wordDictionary.search("b."))
//print(wordDictionary.search("a.d"))
//print(wordDictionary.search("."))

//------------------------------------------------------------------------------------------------


/*
721. Accounts Merge


*/


// Time limit exceeded O(N2)
//func accountsMerge(_ accounts: [[String]]) -> [[String]] {
//    var ans = [[String]](repeating: [], count: accounts.count)
//    var accountsArr = accounts
//    var mergedAccounts = Set<Int>()
//    var i = 0
//    while i < accountsArr.count {
//        if mergedAccounts.contains(i) {
//            i += 1
//            continue
//        }
//        var emails = Set<String>()
//        for e in 1..<accountsArr[i].count {
//            emails.insert(accountsArr[i][e])
//        }
//        var accountUpdated = false
//        for j in i+1..<accountsArr.count {
//            if accountsArr[j].first == accountsArr[i].first && !mergedAccounts.contains(j) {
//                if !emails.intersection(accountsArr[j][1...]).isEmpty {
//                    emails = emails.union(accountsArr[j][1...])
//                    mergedAccounts.insert(j)
//                    accountUpdated = true
//                }
//            }
//        }
//        ans[i] = [accountsArr[i].first!] + Array(emails).sorted()
//        accountsArr[i] = ans[i]
//        if !accountUpdated {
//            i += 1
//        }
//    }
//    return ans.filter({ !$0.isEmpty })
//}

// A solution using Disjoint sets union to solve it in (NLogN)
//class DSU {
//    private var parent: [Int] = []
//
//    init(length: Int) {
//        for i in 0..<length {
//            self.parent.append(i)
//        }
//    }
//
//    func findParent(of u: Int) -> Int {
//        if (parent[u] == u) {
//            return u
//        }
//        parent[u] = findParent(of: parent[u])
//        return parent[u]
//    }
//
//    public func union(_ x: Int, _ y: Int) {
//        parent[findParent(of: x)] = findParent(of: y)
//    }
//}
//
//func accountsMerge(_ accounts: [[String]]) -> [[String]] {
//    let dsu = DSU(length: 10001)
//    var emailToName: [String: String] = [:]
//    var emailToId: [String: Int] = [:]
//    var id = 0
//    for account in accounts {
//        var name = ""
//        for email in account {
//            if name == "" {
//                name = email
//                continue
//            }
//            emailToName[email] = name
//            if emailToId[email] == nil {
//                emailToId[email] = id
//                id += 1
//            }
//            dsu.union(emailToId[account[1]]!, emailToId[email]!)
//        }
//    }
//    var ans = [Int: [String]]()
//    for email in emailToName.keys {
//        let index = dsu.findParent(of: emailToId[email]!)
//        if ans[index] == nil {
//            ans[index] = [email]
//        } else {
//            var emails = ans[index]
//            emails?.append(email)
//            ans[index] = emails
//        }
//    }
//    for (index, collection) in zip(ans.indices, ans.values) {
//        var sorted = collection.sorted()
//        sorted.insert(emailToName[sorted[0]]!, at: 0)
//        ans.values[index] = sorted
//    }
//    return Array(ans.values)
//}
//
//print(accountsMerge([["Kevin","Kevin20@m.co","Kevin9@m.co","Kevin18@m.co","Kevin15@m.co","Kevin17@m.co"],["Kevin","Kevin15@m.co","Kevin19@m.co","Kevin0@m.co","Kevin6@m.co","Kevin13@m.co"],["Kevin","Kevin5@m.co","Kevin11@m.co","Kevin13@m.co","Kevin16@m.co","Kevin2@m.co"],["Kevin","Kevin3@m.co","Kevin4@m.co","Kevin15@m.co","Kevin14@m.co","Kevin16@m.co"],["Kevin","Kevin18@m.co","Kevin8@m.co","Kevin16@m.co","Kevin2@m.co","Kevin8@m.co"],["Kevin","Kevin20@m.co","Kevin10@m.co","Kevin0@m.co","Kevin4@m.co","Kevin7@m.co"],["Kevin","Kevin11@m.co","Kevin17@m.co","Kevin17@m.co","Kevin11@m.co","Kevin13@m.co"],["Kevin","Kevin11@m.co","Kevin13@m.co","Kevin4@m.co","Kevin15@m.co","Kevin11@m.co"],["Kevin","Kevin13@m.co","Kevin5@m.co","Kevin6@m.co","Kevin12@m.co","Kevin14@m.co"],["Kevin","Kevin11@m.co","Kevin16@m.co","Kevin20@m.co","Kevin14@m.co","Kevin4@m.co"],["Kevin","Kevin9@m.co","Kevin2@m.co","Kevin13@m.co","Kevin6@m.co","Kevin3@m.co"],["Kevin","Kevin20@m.co","Kevin7@m.co","Kevin17@m.co","Kevin12@m.co","Kevin0@m.co"],["Kevin","Kevin9@m.co","Kevin9@m.co","Kevin9@m.co","Kevin12@m.co","Kevin18@m.co"],["Kevin","Kevin15@m.co","Kevin10@m.co","Kevin14@m.co","Kevin13@m.co","Kevin20@m.co"],["Kevin","Kevin15@m.co","Kevin18@m.co","Kevin13@m.co","Kevin10@m.co","Kevin19@m.co"],["Kevin","Kevin8@m.co","Kevin15@m.co","Kevin4@m.co","Kevin3@m.co","Kevin10@m.co"],["Kevin","Kevin6@m.co","Kevin9@m.co","Kevin10@m.co","Kevin9@m.co","Kevin6@m.co"],["Kevin","Kevin2@m.co","Kevin8@m.co","Kevin12@m.co","Kevin5@m.co","Kevin9@m.co"],["Kevin","Kevin15@m.co","Kevin20@m.co","Kevin9@m.co","Kevin17@m.co","Kevin4@m.co"],["Kevin","Kevin0@m.co","Kevin4@m.co","Kevin0@m.co","Kevin0@m.co","Kevin1@m.co"]]))


//------------------------------------------------------------------------------------------------

/*
Reverse Vowels of a String
*/

//func reverseVowels(_ s: String) -> String {
//    let vowels: [String: Int] = ["a": 1 , "e": 1, "i": 1, "u": 1 , "o": 1]
//    var vowelsStack = [Character]()
//    let sArr = Array(s)
//    var ans = ""
//    for i in 0..<sArr.count {
//        let char = String(sArr[i]).lowercased()
//        if vowels[char] != nil {
//            vowelsStack.append(sArr[i])
//        }
//    }
//    for i in 0..<sArr.count {
//        let char = String(sArr[i]).lowercased()
//        if vowels[char] != nil {
//            ans.append(vowelsStack.removeLast())
//        } else {
//            ans.append(sArr[i])
//        }
//    }
//    return ans
//}

//------------------------------------------------------------------------------------------------

/*
Find Largest Value in Each Tree Row
*/
//func largestValues(_ root: TreeNode?) -> [Int] {
//  guard let root = root else  {
//    return []
//  }
//  var que = [TreeNode]()
//  que.append(root)
//  var ans = [Int]()
//
//  while !que.isEmpty {
//
//    var nodesLeftInLevel = que.count
//    var mx = Int.min
//    while nodesLeftInLevel > 0 {
//     let node = que.removeFirst()
//      if node.val > mx {
//        mx = node.val
//      }
//      nodesLeftInLevel -= 1
//      if let left = node.left {
//        que.append(left)
//      }
//      if let right = node.right {
//        que.append(right)
//      }
//    }
//   ans.append(mx)
//  }
//    return ans
//}

//------------------------------------------------------------------------------------------------

// FB: 1st interview
//
// Add Binary

//func addBinaryStrings(firstNum: String, secondNum: String) -> String {
//    var fn = Array(firstNum)
//    var sn = Array(secondNum)
//    var iterator = fn.count > sn.count ? fn.count - 1 : sn.count - 1
//    if fn.count > sn.count {
//        sn.insert(contentsOf: Array(repeating: "0", count: iterator + 1 - sn.count), at: 0)
//    } else if fn.count < sn.count {
//        fn.insert(contentsOf: Array(repeating: "0", count: iterator + 1 - fn.count), at: 0)
//    }
//    var carry = 0
//    var ans = ""
//    while iterator >= 0 {
//        let fi = Int(String(fn[iterator])) ?? 0
//        let si = Int(String(sn[iterator])) ?? 0
//        if fi + si + carry == 3  {
//            carry = 1
//            ans.append("1")
//        } else if fi + si + carry == 2 {
//            ans.append("0")
//            carry = 1
//        } else if fi + si + carry == 1 {
//            ans.append("1")
//            carry = 0
//        } else {
//            ans.append("0")
//            carry = 0
//        }
//        iterator -= 1
//    }
//    if carry == 1 {
//        ans.append(String(carry))
//    }
//    return String(ans.reversed())
//}

//print(addBinaryStrings(firstNum: "101", secondNum: "1101"))


//------------------------------------------------------------------------------------------------

//func copyTree(_ root: TreeNode?) -> TreeNode? {
//    guard let root = root else {
//        return nil
//    }
//
//    let copy = TreeNode(root.val)
//    if let left = root.left {
//        copy.left = copyTree(left)
//    }
//    if let right = root.right {
//        copy.right = copyTree(right)
//    }
//    return copy
//}
//
//let n1 = TreeNode(3)
//let n5 = TreeNode(9)
//let n6 = TreeNode(20)
//let n3 = TreeNode(15, n5, n6)
//let root = TreeNode(7, n1, n3)
//var copy = copyTree(root)
//print(copy!)
//------------------------------------------------------------------------------------------------

// ----------------------------------------------------------------------- Preparation for 2nd try -----------------------------------------------------------------------------

// MARK: -ArraysAndStrings
let arraysAndStrings = ArraysAndStrings()
print("lengthOfLongestSubstring")
print(arraysAndStrings.lengthOfLongestSubstringBruteForce("pwwkew")) // Answer 3
print(arraysAndStrings.lengthOfLongestSubstring("pwwkew"))
print(arraysAndStrings.lengthOfLongestSubstringOptimized("pwwkew"))

//------------------------------------------------------------------------------------------------

print("myAtoi")
print(arraysAndStrings.myAtoi("-91283472332")) // -2147483648
print(arraysAndStrings.myAtoiDFA("-91283472332")) // -2147483648

print(arraysAndStrings.myAtoi("2147483648")) // 2147483647
print(arraysAndStrings.myAtoiDFA("2147483648")) // 2147483647

print(arraysAndStrings.myAtoi("+-12")) // 0
print(arraysAndStrings.myAtoiDFA("+-12")) // 0

//------------------------------------------------------------------------------------------------

print("romanToInt")
print(arraysAndStrings.romanToInt("III")) // 3
print(arraysAndStrings.romanToInt("IV")) // 4
print(arraysAndStrings.romanToInt("MCMXCIV")) // 1994

//------------------------------------------------------------------------------------------------

print("threeSum")
print(arraysAndStrings.threeSum([-1,0,1,2,-1,-4]))
print(arraysAndStrings.threeSum([12,-14,-5,12,-2,9,0,9,-3,-3,-14,-6,-4,13,-11,-8,0,5,-7,-6,-10,-13,-7,-14,-3,0,12,5,-8,7,3,-11,0,6,9,13,-8,-6,7,4,6,0,13,-13,-1,9,-13,6,-1,-13,-15,-4,-11,-15,-11,-7,1,-14,13,8,0,2,4,-15,-15,-2,5,-8,7,-11,11,-10,4,1,-15,10,-5,-13,2,1,11,-6,4,-15,-5,8,-7,3,1,-9,-4,-14,0,-15,8,0,-1,-2,7,13,2,-5,11,13,11,11]))

//------------------------------------------------------------------------------------------------

print("RotateImage")
var image = [[1,2,3],[4,5,6],[7,8,9]]
arraysAndStrings.rotate(&image) // Output: [[7,4,1],[8,5,2],[9,6,3]]
print(image)

//------------------------------------------------------------------------------------------------

// Remove Duplicates from Sorted Array
// Time complexity O(N)
// Space Complexity O(1)

//func removeDuplicates(_ nums: inout [Int]) -> Int {
//    if nums.count == 0 {
//        return 0
//    }
//    var i = 0
//    for j in 1..<nums.count {
//        if nums[i] != nums[j] {
//            i += 1
//            nums[i] = nums[j]
//        }
//    }
//    print(nums)
//    return i + 1
//}
//
//var arr1 = [1,1,2]
//var arr2 = [0,0,1,1,1,2,2,3,3,4]
//print(removeDuplicates(&arr1))
//print(removeDuplicates(&arr2))

//------------------------------------------------------------------------------------------------

// Next Permutation
// Time complexity
// Space complexity


//func nextPermutation(_ nums: inout [Int]) {
//    var i = nums.count - 2
//    // Look for a pair satisfy nums[i] > nums[i+1]
//    while (i >= 0 && nums[i] >= nums[i+1]) {
//        i -= 1
//    }
//    // If Exists, then we look for nums[j] > nums[i] where j > i
//    if i >= 0 {
//        var j = nums.count - 1
//        while (nums[j] <= nums[i]) {
//            j -= 1
//        }
//        // If found, swap it with nums[i]
//        nums.swapAt(i, j)
//        debugPrint("\(nums)")
//    }
//    nums[i+1..<nums.count].reverse()
//}
//
//var arr1 = [1,2,3] // [1,3,2]
//print(nextPermutation(&arr1))

//------------------------------------------------------------------------------------------------

// 43. Multiply Strings
//  Time complexity: O(M^2 + M . N)
// Space Complexity: O(M⋅(M+N))
//
//func multiply(_ num1: String, _ num2: String) -> String {
//
//    if num2 == "0" || num1 == "0" {
//        return "0"
//    }
//
//    let num1Reversed = Array(String(num1.reversed()))
//    let num2Reversed = Array(String(num2.reversed()))
//
//    var results: [[Int]] = [[]]
//    results.removeAll()
//
//    for i in 0..<num2Reversed.count {
//        results.append(multiplyOneDigit(num1Reversed, "\(num2Reversed[i])", i))
//    }
//
//    let finalAnswer = sumResults(results: &results).reversed()
//    return String(finalAnswer)
//}
//
//func multiplyOneDigit(_ firstNumber: [String.Element], _ secondNumberDigit: String, _ numZeros: Int) -> [Int] {
//    var currentResult = [Int]()
//    for _ in 0..<numZeros {
//        currentResult.append(0)
//    }
//
//    var carry = 0
//    for i in 0..<firstNumber.count {
//        let firstNumberDigit = "\(firstNumber[i])"
//        let multiplication = Int(secondNumberDigit)! * Int(firstNumberDigit)! + carry
//        carry = multiplication / 10
//        currentResult.append(multiplication % 10)
//    }
//
//    if carry > 0 {
//        currentResult.append(carry)
//    }
//
//    return currentResult
//}
//
//func sumResults(results: inout [[Int]]) -> String {
//    var finalAnswer = results.removeFirst()
//    var newAnswer: [Int] = []
//
//    for result in results {
//        debugPrint("result \(result)")
//        newAnswer.removeAll()
//        var carry = 0
//        var i = 0
//        while (i < result.count || i < finalAnswer.count) {
//            let digit1 = i < result.count ? result[i] : 0
//            let digit2 = i < finalAnswer.count ? finalAnswer[i] : 0
//
//            let sum = Int(digit1) + Int(digit2) + carry
//
//            carry = sum / 10
//            newAnswer.append(sum % 10)
//            i += 1
//        }
//        if carry > 0 {
//            newAnswer.append(carry)
//        }
//        finalAnswer = newAnswer
//    }
//
//    var stringAnswer = ""
//    for digit in finalAnswer {
//        stringAnswer.append("\(digit)")
//    }
//    return stringAnswer
//}
//
//var num1 = "123"
//var num2 = "456"
//print(multiply(num1, num2)) // "56088"


//------------------------------------------------------------------------------------------------

// MARK: - Amazon Assessment
func processLogs(logs: [String], threshold: Int) -> [String] {
    var counter: [String: Int] = [:]
    for log in logs {
        let components = log.components(separatedBy: " ")
        let sender = components[0]
        let receiver  = components[1]
        
        if counter[sender] == nil {
            counter[sender] = 1
        } else {
            counter[sender]! += 1
        }
        
        if sender != receiver {
            if counter[receiver] == nil {
                counter[receiver] = 1
            } else {
                counter[receiver]! += 1
            }
        }
    }
    
    let resultInt = counter.filter({ $0.value >= threshold }).map{  Int32($0.key)! }.sorted(by: <)
    return resultInt.map({"\($0)"})
}
print("processLogs")
print(processLogs(logs: ["1 2 50", "1 7 70", "1 3 20", "2 2 17"], threshold: 2))


func numberOfItems(s: String, startIndices: [Int], endIndices: [Int]) -> [Int] {
    var ans = [Int]()
    let arr = Array(s)
    let n = arr.count
    var dp = Array.init(repeating: 0, count: n)
    var count = 0
    
    for i in 0..<n {
        if arr[i] == "|" {
            dp[i] = count
        } else {
            count += 1
        }
    }
    for i in 0..<startIndices.count {
        var start = startIndices[i] - 1
        var end = endIndices[i] - 1

        while arr[start] != "|" {
            start += 1
        }
        while arr[end] != "|" {
            end -= 1
        }
        if (start < end) {
            ans.append(dp[end] - dp[start])
        } else {
            ans.append(0)
        }
    }
    return ans
}

print("numberOfItems")
print(numberOfItems(s: "*|*|*|", startIndices: [1], endIndices: [6])) // 2


func minimalHeaviestSetA(arr: [Int]) -> [Int] {
    
    var ans = [Int]()
    let sorted = arr.sorted(by: <)
    var sum = 0
    sorted.forEach({ sum += $0 })
    var s = 0

    for i in stride(from: sorted.count - 1, to: 0-1, by: -1) {
        s += sorted[i]
        ans.append(sorted[i])
        
        if sum - s < s {
            break
        }
    }
    ans.sort()
    return ans
}

print("minimalHeaviestSetA")
print(minimalHeaviestSetA(arr: [5,3,2,4,1,2]))

//------------------------------------------------------------------------------------------------

//time O(n), space O(n)
func findTotalPower(power: [Int]) -> Int {
    let MOD = 1000000007
    let n = power.count
    let pOfP = prefixOfPrefixSum(arr: power, n: n)
    let leftSmaller = prevSmaller(arr: power, n: n)
    let rightSmallerOrEqual = nextSmallerOrEqual(arr: power, n: n)
    
    var res = 0
    for i in 0..<n {
        let left = leftSmaller[i]
        let right = rightSmallerOrEqual[i]
        var val = (i-left)*(pOfP[right-1+1]-pOfP[i-1+1])%MOD + MOD - (right-i)*(pOfP[i-1+1]-pOfP[left-1+1<0 ? 0 : left-1+1])%MOD
        val = (power[i]*val)%MOD
        res += val
        res %= MOD
    }
    return res
}
//prefix sum of prefix sum
func prefixOfPrefixSum(arr: [Int], n: Int) -> [Int] {
    var res = Array(repeating: 0, count: n+1)
    var sum = 0
    res[0] = 0
    for i in 1...n {
        sum += arr[i-1]
        sum %= 1000000007
        res[i] = (res[i-1]+sum)%1000000007
    }
    return res
}
            
func prevSmaller(arr: [Int], n: Int) -> [Int] {
    var res = Array(repeating: 0, count: n)
    var st = [Int]()
    res[0] = 0
    for i in 0..<n {
        while !st.isEmpty && arr[st.last!] >= arr[i] {
            _ = st.popLast()
        }
        res[i] = st.isEmpty ? -1 : st.last!
        st.append(i)
    }
    return res
}

func nextSmallerOrEqual(arr: [Int], n: Int) -> [Int] {
    var res = Array(repeating: 0, count: n)
    var st = [Int]()
    res[0] = 0
    for i in stride(from: n-1, to: 0-1, by: -1) {
        while !st.isEmpty && arr[st.last!] > arr[i] {
            _ = st.popLast()
        }
        res[i] = st.isEmpty ? n : st.last!
        st.append(i)
    }
    return res
}

print("findTotalPower")
print(findTotalPower(power: [2,1,3]))


func getMinMoves(plates: [Int]) -> Int {
    let n = plates.count
    var minSeen = Array(repeating: 0, count: n)
    minSeen[n - 1] = plates[n - 1];

    for i in stride(from: n - 2, to: 0-1, by: -1) {
        minSeen[i] = min(minSeen[i+1], plates[i])
    }
    var count = 0
    var maxMoved = Int.max
    for i in 0..<n {
        if plates[i] > maxMoved { // if moved Higher number before then need to move
            count += 1
            
        } else if plates[i] > minSeen[i] { // if there is lower number after then need to move
            count += 1
            maxMoved = plates[i]
        }
    }
    return count
}

print("getMinMoves")
print(getMinMoves(plates: [2,4, 3,1,6]))
