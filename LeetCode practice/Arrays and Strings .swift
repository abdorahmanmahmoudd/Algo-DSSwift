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
    
}
