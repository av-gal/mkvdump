//! The tree module contains helpers for building tree
//! structures from parsed elements
use serde::Serialize;

use crate::{Body, Element, Header, Id};

/// A Master Element that owns its children for diplaying
/// it in an element tree
#[derive(Debug, PartialEq, Serialize)]
pub struct MasterElement {
    #[serde(flatten)]
    header: Header,
    children: Vec<ElementTree>,
}

/// An Element Tree can either be a leaf or a Master
/// element.
#[derive(Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ElementTree {
    /// A Normal Element that represents a leaf in the tree
    Normal(Element),
    /// A Master Element contains more elements.
    Master(MasterElement),
}

impl Id {
    fn can_be_children_of(&self, other: &Id) -> bool {
        !matches!((self, other), (Id::Cluster, Id::Cluster) | (Id::Ebml, _))
    }
}

/// Build element trees from a series of elements
pub fn build_element_trees(elements: &[Element]) -> Vec<ElementTree> {
    let mut trees = Vec::<ElementTree>::new();

    let mut index = 0;
    while index < elements.len() {
        let element = &elements[index];
        match element.body {
            Body::Master => {
                // parse_header() already handles Unknown sizes.
                let mut size_remaining = element.header.body_size.as_u64();

                let mut children = Vec::<Element>::new();
                while size_remaining > 0 {
                    index += 1;

                    if let Some(next_child) = elements.get(index) {
                        if !next_child.header.id.can_be_children_of(&element.header.id) {
                            index -= 1;
                            break;
                        }

                        size_remaining -= if next_child.body == Body::Master {
                            // Master elements' body size should not count in the recursion
                            // as the children would duplicate the size count, so
                            // we only consider the header size on the calculation.
                            next_child.header.size() as u64
                        } else {
                            next_child
                                .size()
                                .expect("Only Master elements can have unknown size")
                        };
                        children.push(next_child.clone());
                    } else {
                        // Elements have ended before reaching the size of the master element
                        break;
                    }
                }
                trees.push(ElementTree::Master(MasterElement {
                    header: element.header.clone(),
                    children: build_element_trees(&children),
                }));
            }
            _ => {
                trees.push(ElementTree::Normal(element.clone()));
            }
        }
        index += 1;
    }
    trees
}

#[cfg(test)]
mod tests {
    use crate::{ebml::VarInt, Unsigned};

    use super::*;

    #[test]
    fn test_build_element_trees() {
        let elements = [
            Element {
                header: Header::new(Id::Ebml, VarInt::new(31)),
                body: Body::Master,
            },
            Element {
                header: Header::new(Id::EbmlVersion, VarInt::new(1)),
                body: Body::Unsigned(Unsigned::Standard(1)),
            },
            Element {
                header: Header::new(Id::EbmlReadVersion, VarInt::new(1)),
                body: Body::Unsigned(Unsigned::Standard(1)),
            },
            Element {
                header: Header::new(Id::EbmlMaxIdLength, VarInt::new(1)),
                body: Body::Unsigned(Unsigned::Standard(4)),
            },
            Element {
                header: Header::new(Id::EbmlMaxSizeLength, VarInt::new(1)),
                body: Body::Unsigned(Unsigned::Standard(8)),
            },
            Element {
                header: Header::new(Id::DocType, VarInt::new(4)),
                body: Body::String("webm".to_string()),
            },
            Element {
                header: Header::new(Id::DocTypeVersion, VarInt::new(1)),
                body: Body::Unsigned(Unsigned::Standard(4)),
            },
            Element {
                header: Header::new(Id::DocTypeReadVersion, VarInt::new(1)),
                body: Body::Unsigned(Unsigned::Standard(2)),
            },
        ];

        let expected = vec![ElementTree::Master(MasterElement {
            header: Header::new(Id::Ebml, VarInt::new(31)),
            children: vec![
                ElementTree::Normal(Element {
                    header: Header::new(Id::EbmlVersion, VarInt::new(1)),
                    body: Body::Unsigned(Unsigned::Standard(1)),
                }),
                ElementTree::Normal(Element {
                    header: Header::new(Id::EbmlReadVersion, VarInt::new(1)),
                    body: Body::Unsigned(Unsigned::Standard(1)),
                }),
                ElementTree::Normal(Element {
                    header: Header::new(Id::EbmlMaxIdLength, VarInt::new(1)),
                    body: Body::Unsigned(Unsigned::Standard(4)),
                }),
                ElementTree::Normal(Element {
                    header: Header::new(Id::EbmlMaxSizeLength, VarInt::new(1)),
                    body: Body::Unsigned(Unsigned::Standard(8)),
                }),
                ElementTree::Normal(Element {
                    header: Header::new(Id::DocType, VarInt::new(4)),
                    body: Body::String("webm".to_string()),
                }),
                ElementTree::Normal(Element {
                    header: Header::new(Id::DocTypeVersion, VarInt::new(1)),
                    body: Body::Unsigned(Unsigned::Standard(4)),
                }),
                ElementTree::Normal(Element {
                    header: Header::new(Id::DocTypeReadVersion, VarInt::new(1)),
                    body: Body::Unsigned(Unsigned::Standard(2)),
                }),
            ],
        })];

        assert_eq!(build_element_trees(&elements), expected);
    }
}
