//
//  Longest Substring Without Repeating Characters.swift
//  LeetCode practice
//
//  Created by Abdelrahman Ali on 14/04/2022.
//  Copyright © 2022 Abdo. All rights reserved.
//

import Foundation

final class ArraysAndStrings {
    
    // MARK: -Longest Substring Without Repeating Characters - Medium
    func lengthOfLongestSubstringBruteForce(_ s: String) -> Int {
        var result = 0
        let stringArray = Array(s)
        for i in 0..<s.count {
            for j in stride(from: i, to: s.count, by: 1) {
                if checkAllUnique(i, j, stringArray) {
                    result = max(result, i - j + 1)
                }
            }
        }
        return result
    }
    private func checkAllUnique(_ start: Int, _ end: Int, _ str: Array<Character>) -> Bool {
        var charsMap: [Character: Int] = [:]
        for i in start...end {
            if charsMap[str[i]] == nil {
                charsMap[str[i]] = 1
            } else {
                return false
            }
        }
        return true
    }
    
    // Idea: Use sliding window
    // time complexity O(2n) = O(n)
    // space complexity O(min(m,n)) = O(k) = the space of the sliding window
    func lengthOfLongestSubstring(_ s: String) -> Int {
        var result = 0
        var charsCounter: [Character: Int] = [:]
        let stringArr = Array(s)
        var left = 0
        var right = 0
        while right < stringArr.count {
            let r = stringArr[right]
            if charsCounter[r] == nil {
                charsCounter[r] = 1
            } else {
                charsCounter[r]! += 1
            }
            
            while charsCounter[r]! > 1 {
                let l = stringArr[left]
                charsCounter[l]! -= 1
                left += 1
            }
            result = max(result, right - left + 1)
            right += 1
        }
        return result
    }
    
    // time complexity O(n)
    // space complexity O(min(m,n)) = O(k) = the space of the sliding window
    func lengthOfLongestSubstringOptimized(_ s: String) -> Int {
        var result = 0
        var charsIndex: [Character: Int] = [:]
        let stringArr = Array(s)
        var left = 0
        var right = 0
        while right < stringArr.count {
            let r = stringArr[right]
            if let charIndex = charsIndex[r], charIndex >= left, charIndex <= right {
                left = charIndex
            }
            result = max(result, right - left + 1)
            right += 1
            charsIndex[r] = right
        }
        return result
    }
    
    // MARK: -String to Integer (atoi) - Medium
    // This brute force approach is sufficient for interviews
    // Time complexity O(n)
    // Space complexity O(1)
    func myAtoi(_ s: String) -> Int {
        let a = Array(s)
        var doubleAns: Double = 0
        var ansSign = Character("+")
        var j = 0
        while j < a.count, a[j] == " " {
            j += 1
        }
        if j < a.count, (a[j] == "-" || a[j] == "+") {
            ansSign = a[j]
            j += 1
        }
        for i in j..<a.count {
            if let integer = Int(String(a[i])) {
                doubleAns = (doubleAns * 10) + Double(integer)
                if doubleAns <= Double(Int32.min) || doubleAns >= Double(Int32.max) {
                    break
                }
            } else  {
                break
            }
        }
        doubleAns = doubleAns * (ansSign == "-" ? -1 : 1)
        if doubleAns <= Double(Int32.min) {
            return Int(Int32.min)
        } else if doubleAns >= Double(Int32.max) {
            return Int(Int32.max)
        }
        return Int(doubleAns)
    }
    
    // Another approach using the Deterministic Finite Automaton (DFA) -> for readability and maintainability
    enum State {
        case q0, q1, q2, qd
    }
    
    class StateMachine {
        var currentState: State = .q0
        var result: Int = 0
        var sign = 1
        
        func toStateQ1(char: Character) {
            sign = (char == "-") ? -1 : 1
            currentState = .q1
        }
        
        func toStateQ2(digit: Int) {
            currentState = .q2
            appendDigit(digit)
        }
        
        func toStateQd() {
            currentState = .qd
        }
        
        func appendDigit(_ digit: Int) {
            if result > Int32.max / 10 || (result == Int32.max / 10 && digit > Int32.max % 10) {
                if sign == 1 {
                    result = Int(Int32.max)
                } else {
                    result = Int(Int32.min)
                    sign = 1
                }
                toStateQd()
            } else {
                result = result * 10 + digit
            }
        }
        
        func isDigit(char: Character) -> Bool {
            return Int(String(char)) != nil
        }
        
        func transition(char: Character) {
            if currentState == .q0 {
                if char == " " {
                    return
                } else if char == "-" || char == "+" {
                    toStateQ1(char: char)
                } else if isDigit(char: char) {
                    toStateQ2(digit: Int(String(char))!)
                } else {
                    toStateQd()
                }
            } else if currentState == .q1 || currentState == .q2 {
                if isDigit(char: char) {
                    toStateQ2(digit: Int(String(char))!)
                } else {
                    toStateQd()
                }
            }
        }
        
        func getResult() -> Int {
            return result * sign
        }
        
        func getState() -> State {
            return currentState
        }
    }
    
    func myAtoiDFA(_ s: String) -> Int {
        let a = Array(s)
        let stateMachine = StateMachine()
        for i in stride(from: 0, to: a.count, by: 1) {
            if stateMachine.getState() == .qd {
                break
            }
            stateMachine.transition(char: a[i])
        }
        return stateMachine.getResult()
    }
    
    // MARK: -Roman to Integer - Easy

    func romanToInt(_ s: String) -> Int {
        var ans = 0
        let romanToInteger: [Character: Int] = ["I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500,  "M": 1000]
        var romanQueue = Array(s.reversed())
        while !romanQueue.isEmpty {
            let r = romanQueue.removeLast()
            if r == "I", let nextR = romanQueue.last, (nextR == "V" || nextR == "X") {
                ans += romanToInteger[nextR]! - romanToInteger[r]!
                romanQueue.removeLast()
            } else if r == "X", let nextR = romanQueue.last, (nextR == "L" || nextR == "C") {
                ans += romanToInteger[nextR]! - romanToInteger[r]!
                romanQueue.removeLast()
            } else if r == "C", let nextR = romanQueue.last, (nextR == "D" || nextR == "M") {
                ans += romanToInteger[nextR]! - romanToInteger[r]!
                romanQueue.removeLast()
            } else {
                ans += romanToInteger[r]!
            }
        }
        return ans
    }
    
    // MARK: -3Sum -
    // Time complexity O(n2) -> Sorting n logN + n2 = n2
    // Space complexity -> depends on the sorting algo (n logN or n)
    // Swift 5 uses the TimSort algorithm for the sorted()

    
    func threeSum(_ nums: [Int]) -> [[Int]] {
        var ans: [[Int]] = []
        let sorted = nums.sorted()
        for i in 0..<sorted.count {
            if i == 0 || sorted[i-1] != sorted[i] {
                twoSumII(sorted, i, &ans)
            }
        }
        return ans
    }
    
    func twoSumII(_ nums: [Int], _ i: Int, _ res: inout [[Int]]) {
        var lo = i + 1
        var hi = nums.count - 1
        while lo < hi {
            let sum = nums[i] + nums[lo] + nums[hi]
            if sum == 0 {
                res.append([nums[i], nums[lo], nums[hi]])
                lo += 1
                hi -= 1
                while lo < hi, nums[lo] == nums[lo - 1] {
                    lo += 1
                }
            } else if sum > 0 {
                hi -= 1
            } else if sum < 0 {
                lo += 1
            }
        }
    }
    
    // MARK: - Rotate Image
    // Time complexity O(N) ->
    // Space complexity -> O(1)
    func rotate(_ matrix: inout [[Int]]) {
        let n = matrix.count
        for i in stride(from: 0, to: (n + 1) / 2, by: 1) {
            for j in stride(from: 0, to: n / 2, by: 1) {
                let temp = matrix[n - 1 - j][i]
                matrix[n - 1 - j][i] = matrix[n - 1 - i][n - 1 - j]
                matrix[n - 1 - i][n - 1 - j] = matrix[j][n - 1 - i]
                matrix[j][n - 1 - i] = matrix[i][j]
                matrix[i][j] = temp
            }
        }
    }
}
