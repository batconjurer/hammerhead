pub mod heuristic;

use std::hash::Hash;
use std::marker::PhantomData;

use rustc_hash::FxHashMap;

use crate::game::space::Role;
use crate::game_tree::{ChildIterator, GameSummary, GameTreeNode, SelectionPolicy};

/// A node in the alpha beta tree that also can iterate over
/// its children statefully.
pub trait InternalNode<N>: Iterator<Item = N> {
    fn node(&self) -> &N;
}

impl InternalNode<GameTreeNode> for ChildIterator {
    fn node(&self) -> &GameTreeNode {
        &self.node
    }
}

/// A node in the game tree
pub trait GameNode: Sized {
    type Convert: InternalNode<Self>;
    fn turn(&self) -> Role;
    fn is_terminal(&self) -> bool;
    fn convert(self) -> Self::Convert;

    fn get_children(&self) -> Vec<Self>;
}

impl GameNode for GameTreeNode {
    type Convert = ChildIterator;

    fn turn(&self) -> Role {
        self.turn
    }

    fn is_terminal(&self) -> bool {
        GameTreeNode::is_terminal(self)
    }

    fn convert(self) -> ChildIterator {
        self.children()
    }

    fn get_children(&self) -> Vec<Self> {
        self.get_children()
    }
}

/// A hashable variant of a game tree node
pub trait ParentNode<'a, N: GameNode + 'a>: Clone + Hash + Eq + From<&'a N> {
    fn turn(&self) -> Role;
}

impl ParentNode<'_, GameTreeNode> for GameSummary {
    fn turn(&self) -> Role {
        self.turn
    }
}

struct AlphaBetaNode<P, N, I>
where
    for<'a> P: ParentNode<'a, N>,
    I: InternalNode<N>,
    N: GameNode<Convert = I>,
{
    parent: P,
    internal_node: I,
    peeked: Option<Peeked<P, N, I>>,
    depth: usize,
    _phantom: PhantomData<N>,
}

/// Indirection to avoid recursive types
struct Peeked<P, N, I>
where
    for<'a> P: ParentNode<'a, N>,
    I: InternalNode<N>,
    N: GameNode<Convert = I>,
{
    parent: P,
    internal_node: I,
    depth: usize,
    _phantom: PhantomData<N>,
}

impl<P, N, I> From<Peeked<P, N, I>> for AlphaBetaNode<P, N, I>
where
    for<'a> P: ParentNode<'a, N>,
    I: InternalNode<N>,
    N: GameNode<Convert = I>,
{
    fn from(peeked: Peeked<P, N, I>) -> Self {
        Self {
            parent: peeked.parent,
            internal_node: peeked.internal_node,
            peeked: None,
            depth: peeked.depth,
            _phantom: Default::default(),
        }
    }
}

impl<P, N, I> AlphaBetaNode<P, N, I>
where
    for<'a> P: ParentNode<'a, N>,
    I: InternalNode<N>,
    N: GameNode<Convert = I>,
{
    fn turn(&self) -> Role {
        self.internal_node.node().turn()
    }

    fn node(&self) -> &N {
        self.internal_node.node()
    }

    /// Get the next child of this node and store it (if it exists)
    fn peek(&mut self) -> bool {
        if self.peeked.is_none() {
            let Some(child) = self.internal_node.next() else {
                return false;
            };
            if self.depth == 0 {
                return false;
            }
            let parent = P::from(self.node());
            self.peeked = Some(Peeked {
                parent: parent.clone(),
                internal_node: child.convert(),
                depth: self.depth - 1,
                _phantom: Default::default(),
            });
        }
        self.peeked.is_some()
    }

    fn next_child(&mut self) -> Option<Self> {
        _ = self.peek();
        self.peeked.take().map(Into::into)
    }

    /// Check if all children in this node has been visited
    fn exhausted(&mut self) -> bool {
        self.node().is_terminal() || !self.peek()
    }

    /// Evaluate this node given the provided heuristic
    fn eval(&self, policy: &impl SelectionPolicy<TreeNode = N>) -> i64 {
        match self.turn() {
            Role::Attacker => policy.eval_attacker(self.node()),
            Role::Defender => policy.eval_defender(self.node()),
        }
    }

    fn is_leaf(&self) -> bool {
        self.depth == 0 || self.node().is_terminal()
    }
}

pub fn alphabeta<P, N, I>(
    root: &N,
    policy: &impl SelectionPolicy<TreeNode = N>,
    depth: usize,
) -> i64
where
    for<'a> P: ParentNode<'a, N>,
    I: InternalNode<N>,
    N: GameNode<Convert = I>,
{
    if depth == 0 {
        return match root.turn() {
            Role::Attacker => policy.eval_attacker(root),
            Role::Defender => policy.eval_defender(root),
        };
    }
    let mut alphas: FxHashMap<P, i64> = FxHashMap::default();
    let mut betas: FxHashMap<P, i64> = FxHashMap::default();
    alphabeta_inner(root, policy, &mut alphas, &mut betas, depth)
}

fn alphabeta_inner<P, N, I>(
    root: &N,
    policy: &impl SelectionPolicy<TreeNode = N>,
    alphas: &mut FxHashMap<P, i64>,
    betas: &mut FxHashMap<P, i64>,
    depth: usize,
) -> i64
where
    for<'a> P: ParentNode<'a, N>,
    I: InternalNode<N>,
    N: GameNode<Convert = I>,
{
    alphas.insert(P::from(root), i64::MIN);
    betas.insert(P::from(root), i64::MAX);

    let mut queue = vec![];
    for child in root.get_children() {
        alphas.insert(P::from(&child), i64::MIN);
        betas.insert(P::from(&child), i64::MAX);
        queue.push(AlphaBetaNode {
            parent: P::from(root),
            internal_node: child.convert(),
            depth: depth - 1,
            peeked: None,
            _phantom: Default::default(),
        });
    }

    // handle the case when the root is also a leaf
    if queue.is_empty() {
        return match root.turn() {
            Role::Attacker => policy.eval_attacker(root),
            Role::Defender => policy.eval_defender(root),
        };
    }
    let mut last_tree_depth = depth;
    while let Some(mut ab_node) = queue.pop() {
        let current_tree_depth = ab_node.depth;
        // we are heading back towards the root after exploring a complete
        // child subtree
        if ab_node.depth > last_tree_depth || ab_node.is_leaf() {
            // update the parents alpha/ beta values based on last explored subtree
            let cutoff = match ab_node.parent.turn() {
                Role::Attacker => {
                    let parent_eval = alphas
                        .get_mut(&ab_node.parent)
                        .expect("A child cannot be visited before its parent");
                    let eval = if ab_node.is_leaf() {
                        ab_node.eval(policy)
                    } else {
                        *betas
                            .get(&P::from(ab_node.node()))
                            .expect("A child evaluation was missing when backtracking up the tree")
                    };
                    // if a full child subtree has been explored or we hit a cutoff,
                    // we can update the parent
                    if *parent_eval >= eval || ab_node.exhausted() {
                        *parent_eval = std::cmp::max(*parent_eval, eval);
                    }
                    *parent_eval >= eval
                }
                Role::Defender => {
                    let parent_eval = betas
                        .get_mut(&ab_node.parent)
                        .expect("A child cannot be visited before its parent");
                    let eval = if ab_node.is_leaf() {
                        ab_node.eval(policy)
                    } else {
                        *alphas
                            .get(&P::from(ab_node.node()))
                            .expect("A child evaluation was missing when backtracking up the tree")
                    };
                    if *parent_eval <= eval || ab_node.exhausted() {
                        *parent_eval = std::cmp::min(*parent_eval, eval);
                    }
                    *parent_eval <= eval
                }
            };
            // we check if all subtrees have been explored. If not, put this node back on the stack
            if !cutoff && !ab_node.is_leaf() && !ab_node.exhausted() {
                queue.push(ab_node);
            } else {
                // we will not visit this node again so it is safe to remove data about it
                let node_key = P::from(ab_node.node());
                alphas.remove(&node_key);
                betas.remove(&node_key);
            }
        } else {
            // we are moving down the tree

            if let Some(child) = ab_node.next_child() {
                // initialize the alpha / beta value for this node in the table if necessary
                let child_key = P::from(child.node());

                let parent_alpha = *alphas
                    .get(&P::from(ab_node.node()))
                    .expect("Cannot visit a child before its parent");
                alphas.insert(child_key.clone(), parent_alpha);

                let parent_beta = *betas
                    .get(&P::from(ab_node.node()))
                    .expect("Cannot visit a child before its parent");
                betas.insert(child_key, parent_beta);

                // re-add this node as it will be visited again on our way back up the tree
                queue.push(ab_node);
                queue.push(child);
            } else {
                // this branch should never really happen
                queue.push(ab_node);
            }
        }
        last_tree_depth = current_tree_depth;
    }
    match root.turn() {
        Role::Attacker => *alphas.get_mut(&P::from(root)).unwrap(),
        Role::Defender => *betas.get_mut(&P::from(root)).unwrap(),
    }
}

#[cfg(test)]
mod test_alphabeta {
    use super::*;
    use std::cell::RefCell;
    use std::cmp::Ordering;
    use std::collections::HashSet;

    #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
    pub struct TestTreeNode {
        level: usize,
        label: usize,
        is_left: bool,
        max_level: usize,
    }

    #[derive(Debug, Copy, Clone)]
    pub struct TestInternal {
        node: TestTreeNode,
        next_child: Option<bool>,
    }

    impl GameNode for TestTreeNode {
        type Convert = TestInternal;

        fn turn(&self) -> Role {
            if self.level & 1 == 1 {
                Role::Defender
            } else {
                Role::Attacker
            }
        }

        fn is_terminal(&self) -> bool {
            self.level == self.max_level
        }

        fn convert(self) -> Self::Convert {
            TestInternal {
                node: self,
                next_child: Some(true),
            }
        }

        fn get_children(&self) -> Vec<Self> {
            if self.is_terminal() {
                vec![]
            } else {
                vec![
                    Self {
                        level: self.level + 1,
                        label: (self.label << 1) + 1,
                        is_left: false,
                        max_level: self.max_level,
                    },
                    Self {
                        level: self.level + 1,
                        label: self.label << 1,
                        is_left: true,
                        max_level: self.max_level,
                    },
                ]
            }
        }
    }

    impl<'a> ParentNode<'a, TestTreeNode> for TestTreeNode {
        fn turn(&self) -> Role {
            if self.level & 1 == 1 {
                Role::Defender
            } else {
                Role::Attacker
            }
        }
    }

    impl From<&TestTreeNode> for TestTreeNode {
        fn from(value: &TestTreeNode) -> Self {
            value.clone()
        }
    }

    impl Iterator for TestInternal {
        type Item = TestTreeNode;

        fn next(&mut self) -> Option<Self::Item> {
            if self.next_child.take()? {
                self.next_child = Some(false);
                Some(TestTreeNode {
                    level: self.node.level + 1,
                    label: self.node.label << 1,
                    is_left: true,
                    max_level: self.node.max_level,
                })
            } else {
                Some(TestTreeNode {
                    level: self.node.level + 1,
                    label: (self.node.label << 1) + 1,
                    is_left: false,
                    max_level: self.node.max_level,
                })
            }
        }
    }

    impl InternalNode<TestTreeNode> for TestInternal {
        fn node(&self) -> &TestTreeNode {
            &self.node
        }
    }

    struct PolicyVector {
        queries: RefCell<HashSet<usize>>,
        evaluations: Vec<i64>,
    }

    impl SelectionPolicy for PolicyVector {
        type TreeNode = TestTreeNode;

        fn eval_attacker(&self, child: &Self::TreeNode) -> i64 {
            let mut queries = self.queries.borrow_mut();
            queries.insert(child.label);
            self.evaluations[child.label]
        }

        fn eval_defender(&self, child: &Self::TreeNode) -> i64 {
            let mut queries = self.queries.borrow_mut();
            queries.insert(child.label);
            self.evaluations[child.label]
        }

        fn compare_children(
            &self,
            _: &Self::TreeNode,
            child1: &Self::TreeNode,
            child2: &Self::TreeNode,
        ) -> Ordering {
            let eval1 = self.evaluations[child1.label];
            let eval2 = self.evaluations[child2.label];
            eval1.cmp(&eval2)
        }
    }

    /// Test basic depth 0 and 1 trees
    #[test]
    fn test_shallow_trees() {
        let root = TestTreeNode {
            level: 0,
            label: 0,
            is_left: false,
            max_level: 0,
        };
        let policy = PolicyVector {
            queries: Default::default(),
            evaluations: vec![10],
        };
        let mut alphas: FxHashMap<TestTreeNode, i64> = FxHashMap::default();
        let mut betas: FxHashMap<TestTreeNode, i64> = FxHashMap::default();
        let res = alphabeta_inner(&root, &policy, &mut alphas, &mut betas, 3);
        assert_eq!(res, 10);
        let root = TestTreeNode {
            level: 0,
            label: 0,
            is_left: false,
            max_level: 1,
        };
        let policy = PolicyVector {
            queries: Default::default(),
            evaluations: vec![1, 2],
        };
        let mut alphas: FxHashMap<TestTreeNode, i64> = FxHashMap::default();
        let mut betas: FxHashMap<TestTreeNode, i64> = FxHashMap::default();
        let res = alphabeta_inner(&root, &policy, &mut alphas, &mut betas, 3);
        assert_eq!(res, 2);
    }

    #[test]
    fn test_pruning() {
        let root = TestTreeNode {
            level: 0,
            label: 0,
            is_left: false,
            max_level: 5,
        };

        let policy = PolicyVector {
            queries: Default::default(),
            evaluations: vec![-1, 3, 5, 7, -6, -4, -8, -9],
        };

        let mut alphas: FxHashMap<TestTreeNode, i64> = FxHashMap::default();
        let mut betas: FxHashMap<TestTreeNode, i64> = FxHashMap::default();
        let res = alphabeta_inner(&root, &policy, &mut alphas, &mut betas, 3);

        assert_eq!(res, 3);
        let mut expected = HashSet::from([0, 1, 2, 4, 5]);
        for queried in policy.queries.borrow().iter() {
            assert!(expected.remove(queried));
        }
        assert!(expected.is_empty());
        let val = alphas.remove(&root).expect("Test failed");
        assert_eq!(val, 3);
        assert!(alphas.is_empty());
        let val = betas.remove(&root).expect("Test failed");
        assert_eq!(val, i64::MAX);
        assert!(betas.is_empty());
    }
}
