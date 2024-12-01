use chrono::prelude::*;
use regex::Regex;
use reqwest::blocking::Client;
use rusqlite::{Connection, Result as SqlResult};
use std::{collections::HashMap, fs, process, env};

fn day_1(input: &str) -> Option<String> {
    let mut first: Vec<u64> = Vec::new();
    let mut second: Vec<u64> = Vec::new();

    for line in input.trim().split('\n') {
        let mut parts = line.splitn(2, char::is_whitespace).map(str::trim);
        if let (Some(first_part), Some(second_part)) = (parts.next(), parts.next()) {
            if let (Ok(first_num), Ok(second_num)) = (first_part.parse(), second_part.parse()) {
                first.push(first_num);
                second.push(second_num);
            }
        }
    }

    // first.sort();
    // second.sort();
    // Some(first.iter()
    //     .zip(second.iter())
    //     .map(|(f, s)| if f > s { f - s } else { s - f })
    //     .sum::<u64>().to_string())

    let mut counter_map: HashMap<u64, u64> = HashMap::new();

    let mut sum: u64 = 0;
    for &item in &first {
        sum += item * (*counter_map.entry(item).or_insert_with(|| {
            second.iter().filter(|&&x| x == item).count() as u64
        }));
    }
    Some(sum.to_string())
}


fn day_2(_: &str) -> Option<String> { None }
fn day_3(_: &str) -> Option<String> { None }
fn day_4(_: &str) -> Option<String> { None }
fn day_5(_: &str) -> Option<String> { None }
fn day_6(_: &str) -> Option<String> { None }
fn day_7(_: &str) -> Option<String> { None }
fn day_8(_: &str) -> Option<String> { None }
fn day_9(_: &str) -> Option<String> { None }
fn day_10(_: &str) -> Option<String> { None }
fn day_11(_: &str) -> Option<String> { None }
fn day_12(_: &str) -> Option<String> { None }
fn day_13(_: &str) -> Option<String> { None }
fn day_14(_: &str) -> Option<String> { None }
fn day_15(_: &str) -> Option<String> { None }
fn day_16(_: &str) -> Option<String> { None }
fn day_17(_: &str) -> Option<String> { None }
fn day_18(_: &str) -> Option<String> { None }
fn day_19(_: &str) -> Option<String> { None }
fn day_20(_: &str) -> Option<String> { None }
fn day_21(_: &str) -> Option<String> { None }
fn day_22(_: &str) -> Option<String> { None }
fn day_23(_: &str) -> Option<String> { None }
fn day_24(_: &str) -> Option<String> { None }
fn day_25(_: &str) -> Option<String> { None }

fn get_session_cookie() -> String {
    let home_dir = env::var("HOME").expect("No $HOME set.");
    let ff_path = std::path::Path::new(&home_dir).join(".mozilla/firefox");
    let mut cookies = Vec::new();

    if let Ok(entries) = fs::read_dir(&ff_path) {
        for entry in entries.flatten() {
            let sub_path = entry.path();
            if sub_path.is_dir() {
                let cookie_file = sub_path.join("cookies.sqlite");

                let temp_file_path = "/tmp/aoccookies.sqlite";
                if fs::copy(&cookie_file, temp_file_path).is_ok() {
                    if let Ok(conn) = Connection::open(temp_file_path) {
                        let mut stmt = conn
                            .prepare("SELECT value FROM moz_cookies WHERE host='.adventofcode.com' AND name='session'")
                            .expect("failed to create SQL query");

                        let session: SqlResult<Vec<String>> = stmt.query_map([], |row| row.get(0))
                            .expect("query execution failed").collect();

                        if let Ok(mut result) = session {
                            cookies.append(&mut result);
                        }
                    }
                }
            }
        }
    }

    cookies.dedup();

    if cookies.is_empty() {
        eprintln!("login at adventofcode.com with a firefox profile.");
        process::exit(1);
    } else if cookies.len() > 1 {
        eprintln!("warning: multiple session cookies found.");
    }

    cookies[0].clone()
}

fn download_input(client: &Client, cookie: &str, day: u32, year: u32) -> String {
    let url = format!("https://adventofcode.com/{}/day/{}/input", year, day);
    client
        .get(&url)
        .header(reqwest::header::COOKIE, cookie)
        .send()
        .unwrap()
        .text()
        .unwrap()
}

fn submit_solution(client: &Client, cookie: &str, day: u32, level: u32, answer: &str, year: u32) -> bool {
    let url = format!("https://adventofcode.com/{}/day/{}/answer", year, day);
    let params = [("level", level.to_string()), ("answer", answer.to_string())];

    println!("Answer: {}", answer);
    println!("Submitting solution for day {}, level {}...\n", day, level);

    let response = client.post(&url)
        .header(reqwest::header::COOKIE, cookie)
        .form(&params)
        .send()
        .unwrap();

    if let Ok(text) = response.text() {
        let re = Regex::new(r"<article><p>(.*?)</p></article>").unwrap();
        let result = if let Some(captures) = re.captures(&text) {
            captures.get(1).map_or("", |m| m.as_str()).trim()
        } else {
            text.trim()
        };

        println!("adventofcode.com says:");

        let completed = "You don't seem to be solving the right level.  Did you already complete it?";
        let locked = "Please don't repeatedly request this endpoint before it unlocks!";
        let wrong = "That's not the right answer";
        let recently = "You gave an answer too recently;";
        let correct = "That's the right answer!";

        let yellow = "\x1b[33m";
        let red = "\x1b[31m";
        let green = "\x1b[32m";
        let reset = "\x1b[0m";

        if result.starts_with(completed) || result.starts_with(locked) {
            println!("{}> {}{}", yellow, result, reset);
            return false;
        } else if result.starts_with(wrong) {
            let i1 = wrong.len();
            let i2 = result.find('.').unwrap_or(result.len());
            let part1 = &result[..i1];
            let part2 = &result[i1..i2];
            let part3 = &result[i2..];

            print!("{}{}{}", red, part1, reset);
            print!("{}{}{}", yellow, part2, reset);
            print!("{}{}{}", red, part3, reset);

            println!();
            return false;
        } else if result.starts_with(recently) {
            let wait_re = Regex::new(r"You have (?:(?P<minutes>\d+)m )?(?P<seconds>\d+)s left to wait").unwrap();
            if let Some(captures) = wait_re.captures(result) {
                let wait_slice = captures.get(0).map_or("", |m| m.as_str());
                let i1 = result.find(wait_slice).unwrap_or(0);
                let i2 = result.rfind(" left to wait").unwrap_or(result.len());

                print!("{}{}{}", yellow, &result[..i1], reset);
                print!("{}{}{}", red, &result[i1..i2], reset);
                print!("{}{}{}", yellow, &result[i2..], reset);
                println!();
            } else {
                println!("{}{}{}", yellow, result, reset);
            }
            return false;
        } else if result.starts_with(correct) {
            println!("{}> {}{}", green, result, reset);
            return true;
        } else {
            println!("{}", result);
            return false;
        }
    }
    false
}

fn solve(client: &Client, year: u32, day: u32) {
    let session_cookie = get_session_cookie();
    let cookie = format!("session={}", session_cookie);

    let input = download_input(client, &cookie, day, year);

    let mut day_functions: HashMap<u32, fn(&str) -> Option<String>> = HashMap::new();
    day_functions.insert(1, day_1);
    day_functions.insert(2, day_2);
    day_functions.insert(3, day_3);
    day_functions.insert(4, day_4);
    day_functions.insert(5, day_5);
    day_functions.insert(6, day_6);
    day_functions.insert(7, day_7);
    day_functions.insert(8, day_8);
    day_functions.insert(9, day_9);
    day_functions.insert(10, day_10);
    day_functions.insert(11, day_11);
    day_functions.insert(12, day_12);
    day_functions.insert(13, day_13);
    day_functions.insert(14, day_14);
    day_functions.insert(15, day_15);
    day_functions.insert(16, day_16);
    day_functions.insert(17, day_17);
    day_functions.insert(18, day_18);
    day_functions.insert(19, day_19);
    day_functions.insert(20, day_20);
    day_functions.insert(21, day_21);
    day_functions.insert(22, day_22);
    day_functions.insert(23, day_23);
    day_functions.insert(24, day_24);
    day_functions.insert(25, day_25);

    if let Some(func) = day_functions.get(&day) {
        match func(&input) {
            Some(result) => {
                println!("Result for day {}: {}", day, result);
                // todo check for "You don't seem to be solving the right level."
                submit_solution(client, &cookie, day, 2, &result, year);
            }
            None => println!("Day {} is not yet implemented.", day),
        }
    } else {
        println!("No function for day {}.", day);
    }
}

fn main() {
    let year = 2024;
    let today = Local::now();
    let day = today.day();

    let client = Client::new();

    solve(&client, year, day);
}
