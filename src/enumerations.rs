// DO NOT EDIT! This file is auto-generated by build.rs.
// Instead, update ebml.xml and ebml_matroska.xml
use crate::ebml::ebml_enumerations;
ebml_enumerations! {
    ChapterTranslateCodec {
        MatroskaScript = 0, original_label = "Matroska Script";
        DvdMenu = 1, original_label = "DVD-menu";
    };
    TrackType {
        Video = 1, original_label = "video";
        Audio = 2, original_label = "audio";
        Complex = 3, original_label = "complex";
        Logo = 16, original_label = "logo";
        Subtitle = 17, original_label = "subtitle";
        Buttons = 18, original_label = "buttons";
        Control = 32, original_label = "control";
        Metadata = 33, original_label = "metadata";
    };
    TrackTranslateCodec {
        MatroskaScript = 0, original_label = "Matroska Script";
        DvdMenu = 1, original_label = "DVD-menu";
    };
    FlagInterlaced {
        Undetermined = 0, original_label = "undetermined";
        Interlaced = 1, original_label = "interlaced";
        Progressive = 2, original_label = "progressive";
    };
    FieldOrder {
        Progressive = 0, original_label = "progressive";
        Tff = 1, original_label = "tff";
        Undetermined = 2, original_label = "undetermined";
        Bff = 6, original_label = "bff";
        BffSwapped = 9, original_label = "bff(swapped)";
        TffSwapped = 14, original_label = "tff(swapped)";
    };
    StereoMode {
        Mono = 0, original_label = "mono";
        SideBySideLeftEyeFirst = 1, original_label = "side by side (left eye first)";
        TopBottomRightEyeIsFirst = 2, original_label = "top - bottom (right eye is first)";
        TopBottomLeftEyeIsFirst = 3, original_label = "top - bottom (left eye is first)";
        CheckboardRightEyeIsFirst = 4, original_label = "checkboard (right eye is first)";
        CheckboardLeftEyeIsFirst = 5, original_label = "checkboard (left eye is first)";
        RowInterleavedRightEyeIsFirst = 6, original_label = "row interleaved (right eye is first)";
        RowInterleavedLeftEyeIsFirst = 7, original_label = "row interleaved (left eye is first)";
        ColumnInterleavedRightEyeIsFirst = 8, original_label = "column interleaved (right eye is first)";
        ColumnInterleavedLeftEyeIsFirst = 9, original_label = "column interleaved (left eye is first)";
        AnaglyphCyanRed = 10, original_label = "anaglyph (cyan/red)";
        SideBySideRightEyeFirst = 11, original_label = "side by side (right eye first)";
        AnaglyphGreenMagenta = 12, original_label = "anaglyph (green/magenta)";
        BothEyesLacedInOneBlockLeftEyeIsFirst = 13, original_label = "both eyes laced in one Block (left eye is first)";
        BothEyesLacedInOneBlockRightEyeIsFirst = 14, original_label = "both eyes laced in one Block (right eye is first)";
    };
    AlphaMode {
        None = 0, original_label = "none";
        Present = 1, original_label = "present";
    };
    OldStereoMode {
        Mono = 0, original_label = "mono";
        RightEye = 1, original_label = "right eye";
        LeftEye = 2, original_label = "left eye";
        BothEyes = 3, original_label = "both eyes";
    };
    DisplayUnit {
        Pixels = 0, original_label = "pixels";
        Centimeters = 1, original_label = "centimeters";
        Inches = 2, original_label = "inches";
        DisplayAspectRatio = 3, original_label = "display aspect ratio";
        Unknown = 4, original_label = "unknown";
    };
    AspectRatioType {
        FreeResizing = 0, original_label = "free resizing";
        KeepAspectRatio = 1, original_label = "keep aspect ratio";
        Fixed = 2, original_label = "fixed";
    };
    MatrixCoefficients {
        Identity = 0, original_label = "Identity";
        ItuRBt709 = 1, original_label = "ITU-R BT.709";
        Unspecified = 2, original_label = "unspecified";
        Reserved1 = 3, original_label = "reserved";
        UsFcc73682 = 4, original_label = "US FCC 73.682";
        ItuRBt470Bg = 5, original_label = "ITU-R BT.470BG";
        Smpte170M = 6, original_label = "SMPTE 170M";
        Smpte240M = 7, original_label = "SMPTE 240M";
        YCoCg = 8, original_label = "YCoCg";
        Bt2020NonConstantLuminance = 9, original_label = "BT2020 Non-constant Luminance";
        Bt2020ConstantLuminance = 10, original_label = "BT2020 Constant Luminance";
        SmpteSt2085 = 11, original_label = "SMPTE ST 2085";
        ChromaDerivedNonConstantLuminance = 12, original_label = "Chroma-derived Non-constant Luminance";
        ChromaDerivedConstantLuminance = 13, original_label = "Chroma-derived Constant Luminance";
        ItuRBt21000 = 14, original_label = "ITU-R BT.2100-0";
    };
    ChromaSitingHorz {
        Unspecified = 0, original_label = "unspecified";
        LeftCollocated = 1, original_label = "left collocated";
        Half = 2, original_label = "half";
    };
    ChromaSitingVert {
        Unspecified = 0, original_label = "unspecified";
        TopCollocated = 1, original_label = "top collocated";
        Half = 2, original_label = "half";
    };
    Range {
        Unspecified = 0, original_label = "unspecified";
        BroadcastRange = 1, original_label = "broadcast range";
        FullRangeNoClipping = 2, original_label = "full range (no clipping)";
        DefinedByMatrixCoefficientsTransferCharacteristics = 3, original_label = "defined by MatrixCoefficients / TransferCharacteristics";
    };
    TransferCharacteristics {
        Reserved1 = 0, original_label = "reserved";
        ItuRBt709 = 1, original_label = "ITU-R BT.709";
        Unspecified = 2, original_label = "unspecified";
        Reserved2 = 3, original_label = "reserved";
        Gamma22CurveBt470M = 4, original_label = "Gamma 2.2 curve - BT.470M";
        Gamma28CurveBt470Bg = 5, original_label = "Gamma 2.8 curve - BT.470BG";
        Smpte170M = 6, original_label = "SMPTE 170M";
        Smpte240M = 7, original_label = "SMPTE 240M";
        Linear = 8, original_label = "Linear";
        Log = 9, original_label = "Log";
        LogSqrt = 10, original_label = "Log Sqrt";
        Iec6196624 = 11, original_label = "IEC 61966-2-4";
        ItuRBt1361ExtendedColourGamut = 12, original_label = "ITU-R BT.1361 Extended Colour Gamut";
        Iec6196621 = 13, original_label = "IEC 61966-2-1";
        ItuRBt202010Bit = 14, original_label = "ITU-R BT.2020 10 bit";
        ItuRBt202012Bit = 15, original_label = "ITU-R BT.2020 12 bit";
        ItuRBt2100PerceptualQuantization = 16, original_label = "ITU-R BT.2100 Perceptual Quantization";
        SmpteSt4281 = 17, original_label = "SMPTE ST 428-1";
        AribStdB67Hlg = 18, original_label = "ARIB STD-B67 (HLG)";
    };
    Primaries {
        Reserved1 = 0, original_label = "reserved";
        ItuRBt709 = 1, original_label = "ITU-R BT.709";
        Unspecified = 2, original_label = "unspecified";
        Reserved2 = 3, original_label = "reserved";
        ItuRBt470M = 4, original_label = "ITU-R BT.470M";
        ItuRBt470BgBt601625 = 5, original_label = "ITU-R BT.470BG - BT.601 625";
        ItuRBt601525Smpte170M = 6, original_label = "ITU-R BT.601 525 - SMPTE 170M";
        Smpte240M = 7, original_label = "SMPTE 240M";
        Film = 8, original_label = "FILM";
        ItuRBt2020 = 9, original_label = "ITU-R BT.2020";
        SmpteSt4281 = 10, original_label = "SMPTE ST 428-1";
        SmpteRp4322 = 11, original_label = "SMPTE RP 432-2";
        SmpteEg4322 = 12, original_label = "SMPTE EG 432-2";
        EbuTech3213EJedecP22Phosphors = 22, original_label = "EBU Tech. 3213-E - JEDEC P22 phosphors";
    };
    ProjectionType {
        Rectangular = 0, original_label = "rectangular";
        Equirectangular = 1, original_label = "equirectangular";
        Cubemap = 2, original_label = "cubemap";
        Mesh = 3, original_label = "mesh";
    };
    TrackPlaneType {
        LeftEye = 0, original_label = "left eye";
        RightEye = 1, original_label = "right eye";
        Background = 2, original_label = "background";
    };
    ContentEncodingScope {
        Block = 1, original_label = "Block";
        Private = 2, original_label = "Private";
        Next = 4, original_label = "Next";
    };
    ContentEncodingType {
        Compression = 0, original_label = "Compression";
        Encryption = 1, original_label = "Encryption";
    };
    ContentCompAlgo {
        Zlib = 0, original_label = "zlib";
        Bzlib = 1, original_label = "bzlib";
        Lzo1X = 2, original_label = "lzo1x";
        HeaderStripping = 3, original_label = "Header Stripping";
    };
    ContentEncAlgo {
        NotEncrypted = 0, original_label = "Not encrypted";
        Des = 1, original_label = "DES";
        TripleDes = 2, original_label = "3DES";
        Twofish = 3, original_label = "Twofish";
        Blowfish = 4, original_label = "Blowfish";
        Aes = 5, original_label = "AES";
    };
    AesSettingsCipherMode {
        AesCtr = 1, original_label = "AES-CTR";
        AesCbc = 2, original_label = "AES-CBC";
    };
    ContentSigAlgo {
        NotSigned = 0, original_label = "Not signed";
        Rsa = 1, original_label = "RSA";
    };
    ContentSigHashAlgo {
        NotSigned = 0, original_label = "Not signed";
        Sha1160 = 1, original_label = "SHA1-160";
        Md5 = 2, original_label = "MD5";
    };
    ChapProcessTime {
        DuringTheWholeChapter = 0, original_label = "during the whole chapter";
        BeforeStartingPlayback = 1, original_label = "before starting playback";
        AfterPlaybackOfTheChapter = 2, original_label = "after playback of the chapter";
    };
    TargetTypeValue {
        Collection = 70, original_label = "COLLECTION";
        EditionIssueVolumeOpusSeasonSequel = 60, original_label = "EDITION / ISSUE / VOLUME / OPUS / SEASON / SEQUEL";
        AlbumOperaConcertMovieEpisode = 50, original_label = "ALBUM / OPERA / CONCERT / MOVIE / EPISODE";
        PartSession = 40, original_label = "PART / SESSION";
        TrackSongChapter = 30, original_label = "TRACK / SONG / CHAPTER";
        SubtrackPartMovementScene = 20, original_label = "SUBTRACK / PART / MOVEMENT / SCENE";
        Shot = 10, original_label = "SHOT";
    };
}
