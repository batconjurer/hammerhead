use std::collections::{HashMap, HashSet, VecDeque};

use crate::game::board::Board;
use crate::game::space::{EXIT_SQUARES, Role, Space, Square, SquareMap};
use rayon::iter::Either;

/// Given a board state, we find the maximum flow
/// from the king's position to each of the four corners
/// ignoring the defenders. This is a value between 0 and 8
/// inclusive.
pub fn escape_routes(board: &Board) -> u8 {
    let Some(king) = board.find_the_king() else {
        return 0;
    };

    EXIT_SQUARES
        .into_iter()
        .map(|c| edmonds_karp(board, king, c))
        .sum()
}

fn get_neighbors<F>(board: &Board, square: Square, predicate: F) -> [Option<Square>; 4]
where
    F: Fn(&Board, Square) -> bool,
{
    let mut neighbors = [None; 4];
    if let Some(sq) = square.up() {
        if predicate(board, sq) {
            neighbors[0] = Some(sq);
        }
    }
    if let Some(sq) = square.left() {
        if predicate(board, sq) {
            neighbors[1] = Some(sq);
        }
    }
    if let Some(sq) = square.right() {
        if predicate(board, sq) {
            neighbors[2] = Some(sq);
        }
    }
    if let Some(sq) = square.down() {
        if predicate(board, sq) {
            neighbors[3] = Some(sq);
        }
    }
    neighbors
}

type Predecessor = SquareMap<Square>;

#[derive(Debug)]
struct EdgeFlows {
    flows: [[i64; 121]; 121],
}

impl Default for EdgeFlows {
    fn default() -> Self {
        Self {
            flows: [[0; 121]; 121],
        }
    }
}

impl EdgeFlows {
    fn get(&self, [f, s]: &[Square; 2]) -> i64 {
        let f_ix = f.y * 11 + f.x;
        let s_ix = s.y * 11 + s.x;
        self.flows[f_ix][s_ix]
    }

    fn insert(&mut self, [f, s]: [Square; 2], flow: i64) {
        let f_ix = f.y * 11 + f.x;
        let s_ix = s.y * 11 + s.x;
        self.flows[f_ix][s_ix] = flow;
    }
}

fn edmonds_karp(board: &Board, king: Square, corner: Square) -> u8 {
    let mut flow_total = 0u8;
    let mut flow = EdgeFlows::default();

    loop {
        let mut queue = VecDeque::from([king]);
        let mut pred = Predecessor::default();
        // look for a path from the king to the corner in the residual graph
        while let Some(square) = queue.pop_front() {
            if pred.contains_key(&corner) {
                break;
            }
            for n in get_neighbors(board, square, |board, sq| {
                matches!(
                    board.get(&sq),
                    Space::Empty | Space::Occupied(Role::Defender),
                )
            })
            .into_iter()
            .flatten()
            {
                if !pred.contains_key(&n) && 1i64 > flow.get(&[square, n]) {
                    pred.insert(n, square);
                    queue.push_back(n);
                }
            }
        }

        // an augmenting path was found
        if pred.contains_key(&corner) {
            // we want to see how much flow we can send along this path
            let mut delta_flow = u8::MAX;
            let mut cur = corner;

            while let Some(p) = pred.get(&cur) {
                let current_flow = flow.get(&[*p, cur]);
                delta_flow = std::cmp::min(delta_flow, (1 - current_flow) as u8);
                cur = *p;
            }
            // update the flow by the computed amount
            let mut cur = corner;
            while let Some(p) = pred.get(&cur) {
                let current_flow = flow.get(&[*p, cur]);
                flow.insert([*p, cur], current_flow + delta_flow as i64);
                let rev_flow = flow.get(&[cur, *p]);
                flow.insert([cur, *p], rev_flow - delta_flow as i64);
                cur = *p;
            }
            flow_total += delta_flow;
        } else {
            break;
        }
        if flow_total == 2 {
            return 2;
        }
    }
    flow_total
}

/// Given a board state, we find out the shortest path from the king to an
/// escape square if any exists.
#[allow(dead_code)]
pub fn shortest_escape(board: &Board) -> Option<u8> {
    let king = board.find_the_king()?;
    let mut queue = VecDeque::from([king]);
    let mut pred = HashMap::<Square, Square>::new();
    let mut escape = None;
    // look for a path from the king to a corner
    while let Some(square) = queue.pop_front() {
        for n in get_neighbors(board, square, |board, sq| {
            matches!(board.get(&sq), Space::Empty)
        })
        .into_iter()
        .flatten()
        {
            if let std::collections::hash_map::Entry::Vacant(e) = pred.entry(n) {
                e.insert(square);
                if EXIT_SQUARES.contains(&n) {
                    escape = Some(n);
                    break;
                } else {
                    queue.push_back(n);
                }
            }
        }
        if escape.is_some() {
            break;
        }
    }
    let mut dist = 0u8;
    let mut cur = escape?;
    while let Some(p) = pred.get(&cur) {
        dist += 1;
        cur = *p
    }
    Some(dist)
}

/// Given a board state, we find out the fewest number of moves
/// the king must make to an escape, if any exists. This corresponds
/// the path with fewest "turns" or "corners" to an exit square.
pub fn fewest_turns_to_escape(board: &Board) -> Option<u8> {
    let mut current_turns = 1u8;
    let king = board.find_the_king()?;
    let mut visited = HashSet::from([king]);
    let mut starts = HashSet::from([king]);
    loop {
        let mut next_starts = HashSet::new();
        for cursor in &starts {
            // find all squares reachable in a string line from this square
            match advance_linearly(*cursor, board, &mut visited, current_turns) {
                Either::Left(found) => next_starts.extend(found),
                Either::Right(res) => return Some(res),
            }
        }
        if next_starts.is_empty() {
            // we visited all squares reachable from the king without finding an exit square
            return None;
        } else {
            std::mem::swap(&mut starts, &mut next_starts);
        }
        current_turns += 1;
    }
}

fn advance_linearly(
    cursor: Square,
    board: &Board,
    visited: &mut HashSet<Square>,
    current_turns: u8,
) -> Either<Vec<Square>, u8> {
    let mut next_starts = vec![];
    let mut left_cursor = cursor;
    while let Some(next) = left_cursor.left() {
        if EXIT_SQUARES.contains(&next) {
            return Either::Right(current_turns);
        }
        if board.is_occupied(&next) {
            break;
        }
        if !visited.contains(&next) {
            next_starts.push(next);
            visited.insert(next);
        }
        left_cursor = next;
    }
    let mut right_cursor = cursor;
    while let Some(next) = right_cursor.right() {
        if EXIT_SQUARES.contains(&next) {
            return Either::Right(current_turns);
        }
        if board.is_occupied(&next) {
            break;
        }
        if !visited.contains(&next) {
            next_starts.push(next);
            visited.insert(next);
        }
        right_cursor = next;
    }
    let mut up_cursor = cursor;
    while let Some(next) = up_cursor.up() {
        if EXIT_SQUARES.contains(&next) {
            return Either::Right(current_turns);
        }
        if board.is_occupied(&next) {
            break;
        }
        if !visited.contains(&next) {
            next_starts.push(next);
            visited.insert(next);
        }
        up_cursor = next;
    }
    let mut down_cursor = cursor;
    while let Some(next) = down_cursor.down() {
        if EXIT_SQUARES.contains(&next) {
            return Either::Right(current_turns);
        }
        if board.is_occupied(&next) {
            break;
        }
        if !visited.contains(&next) {
            next_starts.push(next);
            visited.insert(next);
        }
        down_cursor = next;
    }
    Either::Left(next_starts)
}

#[cfg(test)]
mod test_heuristics {
    use super::*;

    #[test]
    fn test_flows() {
        let board = Board::default();
        let flow = escape_routes(&board);
        assert_eq!(flow, 8);
        let board = [
            ".........O.",
            "..........O",
            "OO.........",
            "K..........",
            "...........",
            "...........",
            "...........",
            "...........",
            "...........",
            "O.........O",
            ".O.......O.",
        ];
        let board = Board::try_from(board).expect("Test failed");
        let king = board.find_the_king().expect("Test failed");
        let flow = edmonds_karp(&board, king, Square { x: 0, y: 0 });
        assert_eq!(flow, 2);
        let flow = escape_routes(&board);
        assert_eq!(flow, 2);
        let board = [
            ".O.......O.",
            "..........O",
            "OO.........",
            "K..........",
            "...........",
            "...........",
            "...........",
            "...........",
            "...........",
            "O.........O",
            ".O.......O.",
        ];
        let board = Board::try_from(board).expect("Test failed");
        let king = board.find_the_king().expect("Test failed");
        let flow = edmonds_karp(&board, king, Square { x: 0, y: 0 });
        assert_eq!(flow, 1);
        let flow = escape_routes(&board);
        assert_eq!(flow, 1);
        let board = [
            ".O.......O.",
            ".O........O",
            "OO.........",
            "K..........",
            "...........",
            "...........",
            "...........",
            "...........",
            "...........",
            "O.........O",
            ".O.......O.",
        ];
        let board = Board::try_from(board).expect("Test failed");
        let flow = escape_routes(&board);
        assert_eq!(flow, 0);

        let board = [
            ".O......O..",
            "...........",
            "OO.........",
            "K..........",
            "...........",
            "...........",
            "...........",
            "...........",
            "...........",
            "..........O",
            ".........O.",
        ];
        let board = Board::try_from(board).expect("Test failed");
        let flow = escape_routes(&board);
        assert_eq!(flow, 5);
    }

    #[test]
    fn test_escape_distance() {
        let board = Board::default();
        assert!(shortest_escape(&board).is_none());
        let board = [
            ".O.......O.",
            ".O........O",
            "OO.........",
            "K..........",
            "...........",
            "...........",
            "...........",
            "...........",
            "...........",
            "O.........O",
            ".O.......O.",
        ];
        let board = Board::try_from(board).expect("Test failed");
        assert!(shortest_escape(&board).is_none());
        let board = [
            ".........O.",
            ".O........O",
            "OO.........",
            "K..........",
            "...........",
            "...........",
            "...........",
            "...........",
            "...........",
            "O.........O",
            ".O.......O.",
        ];
        let board = Board::try_from(board).expect("Test failed");
        let dist = shortest_escape(&board).expect("Test failed");
        assert_eq!(dist, 7);
        let board = [
            ".O.......O.",
            "..........O",
            "OO.........",
            "...........",
            ".....K.....",
            "...........",
            "...........",
            "...........",
            "...........",
            "O.........O",
            ".O.......O.",
        ];
        let board = Board::try_from(board).expect("Test failed");
        let dist = shortest_escape(&board).expect("Test failed");
        assert_eq!(dist, 9);
        let board = [
            ".O.......O.",
            "..........O",
            "OO.........",
            "....XX.....",
            "...X.K.....",
            "...........",
            "...........",
            "...........",
            "...........",
            "O.........O",
            ".O.......O.",
        ];
        let board = Board::try_from(board).expect("Test failed");
        let dist = shortest_escape(&board).expect("Test failed");
        assert_eq!(dist, 11);
    }

    #[test]
    fn test_advance_linearly() {
        let board = Board::default();
        let king = board.find_the_king().expect("Test failed");
        let mut visited = HashSet::from([king]);
        let starts = advance_linearly(king, &board, &mut visited, 0).expect_left("Test failed");
        assert!(starts.is_empty());
        assert!(fewest_turns_to_escape(&board).is_none());
        let board = [
            ".O.......O.",
            "...........",
            "OO...X.....",
            "....X.X....",
            "...X.K.O...",
            "...........",
            ".....O.....",
            "...........",
            "...........",
            "O.........O",
            ".O.......O.",
        ];
        let board = Board::try_from(board).expect("Test failed");
        let king = board.find_the_king().expect("Test failed");
        let mut visited = HashSet::from([king]);
        let starts = advance_linearly(king, &board, &mut visited, 0)
            .expect_left("Test failed")
            .into_iter()
            .collect::<HashSet<_>>();

        let expected = HashSet::from([
            Square { x: 4, y: 4 },
            Square { x: 6, y: 4 },
            Square { x: 5, y: 5 },
            Square { x: 5, y: 3 },
        ]);
        assert_eq!(starts, expected);
        let turns = fewest_turns_to_escape(&board).expect("Test failed");
        assert_eq!(turns, 3);
        let board = [
            ".O.......O.",
            "...........",
            "...........",
            "OO...X.....",
            "....X.X....",
            "...X.K.O...",
            "....XO..O..",
            "......O....",
            "...........",
            "O.........O",
            ".O.......O.",
        ];
        let board = Board::try_from(board).expect("Test failed");
        let turns = fewest_turns_to_escape(&board).expect("Test failed");
        assert_eq!(turns, 6);
    }
}
