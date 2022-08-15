use std::collections::HashSet;
use std::fs::File;
use std::io::Write;

use convert_case::{Case, Casing};
use serde::{Deserialize, Serialize};

const EBML_XML: &str = include_str!("ebml.xml");
const EBML_MATROSKA_XML: &str = include_str!("ebml_matroska.xml");

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct EBMLSchema {
    #[serde(rename(deserialize = "$value"))]
    elements: Vec<Element>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Element {
    name: String,
    path: String,
    id: String,
    #[serde(rename(deserialize = "type"))]
    variant: String,
    #[serde(rename(deserialize = "$value"))]
    details: Option<Vec<ElementDetail>>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ElementDetail {
    Documentation(Documentation),
    Extension(Extension),
    Restriction(Restriction),
    ImplementationNote(ImplementationNote),
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Documentation {
    #[serde(rename(deserialize = "$value"))]
    text: String,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Extension {
    webm: Option<bool>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Restriction {
    #[serde(rename(deserialize = "$value"))]
    enums: Vec<Enum>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Enum {
    value: String,
    label: String,
    #[serde(rename(deserialize = "$value"))]
    documentation: Option<Vec<Documentation>>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct ImplementationNote {
    #[serde(rename(deserialize = "$value"))]
    text: String,
}

fn get_elements() -> Vec<Element> {
    let ebml_schema: EBMLSchema = serde_xml_rs::from_str(EBML_XML).unwrap();
    let ebml_matroska_schema: EBMLSchema = serde_xml_rs::from_str(EBML_MATROSKA_XML).unwrap();

    // Ignoring Matroska overrides of EBML elements
    let mut known_elements = HashSet::<String>::new();
    let mut elements = Vec::<Element>::new();
    for element in ebml_schema
        .elements
        .into_iter()
        .chain(ebml_matroska_schema.elements.into_iter())
    {
        if known_elements.get(&element.name).is_none() {
            known_elements.insert(element.name.clone());
            elements.push(element);
        }
    }

    // Pre-format names and variants
    elements.iter_mut().for_each(|e| {
        e.variant = variant_to_enum_literal(&e.variant).to_string();
    });

    elements
}

fn variant_to_enum_literal(variant: &str) -> &str {
    match variant {
        "master" => "Master",
        "uinteger" => "Unsigned",
        "integer" => "Signed",
        "string" => "String",
        "binary" => "Binary",
        "utf-8" => "Utf8",
        "date" => "Date",
        "float" => "Float",
        _ => panic!("Variant not expected: {}", variant),
    }
}

fn apply_label_quirks(label: &str, reserved_index: &mut i32) -> String {
    let mut label = label
        .replace(|c: char| !c.is_ascii_alphanumeric(), " ")
        .to_case(Case::Pascal);

    // Hack because identifiers can't start with a number
    if label == "3Des" {
        label = "TripleDes".to_string();
    }
    // "Reserved" sometimes repeats in enums
    else if label == "Reserved" {
        label = format!("Reserved{}", reserved_index);
        *reserved_index += 1;
    }

    label
}

fn create_elements_file(elements: &[Element]) -> std::io::Result<()> {
    let mut file = File::create("src/elements.rs")?;

    writeln!(
        file,
        "// DO NOT EDIT! This file is auto-generated by build.rs."
    )?;
    writeln!(file, "// Instead, update ebml.xml and ebml_matroska.xml")?;
    writeln!(file, "use crate::ebml::ebml_elements;")?;
    writeln!(file, "ebml_elements! {{")?;

    for Element {
        name,
        id,
        variant,
        path: _,
        details: _,
    } in elements
    {
        let enum_name = name.to_case(Case::Pascal);
        writeln!(
            file,
            "    name = {enum_name}, original_name = \"{name}\", id = {id}, variant = {variant};"
        )?;
    }
    writeln!(file, "}}")?;

    Ok(())
}

fn create_enumerations_file(elements: &[Element]) -> std::io::Result<()> {
    let mut file = File::create("src/enumerations.rs")?;

    writeln!(
        file,
        "// DO NOT EDIT! This file is auto-generated by build.rs."
    )?;
    writeln!(file, "// Instead, update ebml.xml and ebml_matroska.xml")?;
    writeln!(file, "use crate::ebml::ebml_enumerations;")?;
    writeln!(file, "ebml_enumerations! {{")?;

    for element in elements {
        let mut reserved_index = 1;
        if element.variant != "Unsigned" {
            continue;
        }
        let enum_name = element.name.to_case(Case::Pascal);
        if let Some(details) = &element.details {
            for detail in details {
                if let ElementDetail::Restriction(restriction) = detail {
                    writeln!(file, "    {} {{", enum_name)?;
                    for enumeration in &restriction.enums {
                        let label = apply_label_quirks(&enumeration.label, &mut reserved_index);
                        writeln!(
                            file,
                            "        {} = {}, original_label = \"{}\";",
                            label, enumeration.value, enumeration.label
                        )?;
                    }
                    writeln!(file, "    }};")?;
                }
            }
        }
    }
    writeln!(file, "}}")?;

    Ok(())
}

fn main() -> std::io::Result<()> {
    println!("cargo:rerun-if-changed=ebml.xml");
    println!("cargo:rerun-if-changed=ebml_matroska.xml");

    let elements = get_elements();
    create_elements_file(&elements)?;
    create_enumerations_file(&elements)?;

    Ok(())
}
