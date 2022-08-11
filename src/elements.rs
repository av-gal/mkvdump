// DO NOT EDIT! This file is auto-generated by build.rs.
// Instead, update ebml.xml and ebml_matroska.xml
use crate::ebml::ebml_elements;
ebml_elements! {
    name = Ebml, id = 0x1A45DFA3, variant = Master;
    name = EbmlVersion, id = 0x4286, variant = Unsigned;
    name = EbmlReadVersion, id = 0x42F7, variant = Unsigned;
    name = EbmlMaxIdLength, id = 0x42F2, variant = Unsigned;
    name = EbmlMaxSizeLength, id = 0x42F3, variant = Unsigned;
    name = DocType, id = 0x4282, variant = String;
    name = DocTypeVersion, id = 0x4287, variant = Unsigned;
    name = DocTypeReadVersion, id = 0x4285, variant = Unsigned;
    name = DocTypeExtension, id = 0x4281, variant = Master;
    name = DocTypeExtensionName, id = 0x4283, variant = String;
    name = DocTypeExtensionVersion, id = 0x4284, variant = Unsigned;
    name = Void, id = 0xEC, variant = Binary;
    name = Crc32, id = 0xBF, variant = Binary;
    name = Segment, id = 0x18538067, variant = Master;
    name = SeekHead, id = 0x114D9B74, variant = Master;
    name = Seek, id = 0x4DBB, variant = Master;
    name = SeekId, id = 0x53AB, variant = Binary;
    name = SeekPosition, id = 0x53AC, variant = Unsigned;
    name = Info, id = 0x1549A966, variant = Master;
    name = SegmentUuid, id = 0x73A4, variant = Binary;
    name = SegmentFilename, id = 0x7384, variant = Utf8;
    name = PrevUuid, id = 0x3CB923, variant = Binary;
    name = PrevFilename, id = 0x3C83AB, variant = Utf8;
    name = NextUuid, id = 0x3EB923, variant = Binary;
    name = NextFilename, id = 0x3E83BB, variant = Utf8;
    name = SegmentFamily, id = 0x4444, variant = Binary;
    name = ChapterTranslate, id = 0x6924, variant = Master;
    name = ChapterTranslateId, id = 0x69A5, variant = Binary;
    name = ChapterTranslateCodec, id = 0x69BF, variant = Unsigned;
    name = ChapterTranslateEditionUid, id = 0x69FC, variant = Unsigned;
    name = TimestampScale, id = 0x2AD7B1, variant = Unsigned;
    name = Duration, id = 0x4489, variant = Float;
    name = DateUtc, id = 0x4461, variant = Date;
    name = Title, id = 0x7BA9, variant = Utf8;
    name = MuxingApp, id = 0x4D80, variant = Utf8;
    name = WritingApp, id = 0x5741, variant = Utf8;
    name = Cluster, id = 0x1F43B675, variant = Master;
    name = Timestamp, id = 0xE7, variant = Unsigned;
    name = SilentTracks, id = 0x5854, variant = Master;
    name = SilentTrackNumber, id = 0x58D7, variant = Unsigned;
    name = Position, id = 0xA7, variant = Unsigned;
    name = PrevSize, id = 0xAB, variant = Unsigned;
    name = SimpleBlock, id = 0xA3, variant = Binary;
    name = BlockGroup, id = 0xA0, variant = Master;
    name = Block, id = 0xA1, variant = Binary;
    name = BlockVirtual, id = 0xA2, variant = Binary;
    name = BlockAdditions, id = 0x75A1, variant = Master;
    name = BlockMore, id = 0xA6, variant = Master;
    name = BlockAdditional, id = 0xA5, variant = Binary;
    name = BlockAddId, id = 0xEE, variant = Unsigned;
    name = BlockDuration, id = 0x9B, variant = Unsigned;
    name = ReferencePriority, id = 0xFA, variant = Unsigned;
    name = ReferenceBlock, id = 0xFB, variant = Signed;
    name = ReferenceVirtual, id = 0xFD, variant = Signed;
    name = CodecState, id = 0xA4, variant = Binary;
    name = DiscardPadding, id = 0x75A2, variant = Signed;
    name = Slices, id = 0x8E, variant = Master;
    name = TimeSlice, id = 0xE8, variant = Master;
    name = LaceNumber, id = 0xCC, variant = Unsigned;
    name = FrameNumber, id = 0xCD, variant = Unsigned;
    name = BlockAdditionId, id = 0xCB, variant = Unsigned;
    name = Delay, id = 0xCE, variant = Unsigned;
    name = SliceDuration, id = 0xCF, variant = Unsigned;
    name = ReferenceFrame, id = 0xC8, variant = Master;
    name = ReferenceOffset, id = 0xC9, variant = Unsigned;
    name = ReferenceTimestamp, id = 0xCA, variant = Unsigned;
    name = EncryptedBlock, id = 0xAF, variant = Binary;
    name = Tracks, id = 0x1654AE6B, variant = Master;
    name = TrackEntry, id = 0xAE, variant = Master;
    name = TrackNumber, id = 0xD7, variant = Unsigned;
    name = TrackUid, id = 0x73C5, variant = Unsigned;
    name = TrackType, id = 0x83, variant = Unsigned;
    name = FlagEnabled, id = 0xB9, variant = Unsigned;
    name = FlagDefault, id = 0x88, variant = Unsigned;
    name = FlagForced, id = 0x55AA, variant = Unsigned;
    name = FlagHearingImpaired, id = 0x55AB, variant = Unsigned;
    name = FlagVisualImpaired, id = 0x55AC, variant = Unsigned;
    name = FlagTextDescriptions, id = 0x55AD, variant = Unsigned;
    name = FlagOriginal, id = 0x55AE, variant = Unsigned;
    name = FlagCommentary, id = 0x55AF, variant = Unsigned;
    name = FlagLacing, id = 0x9C, variant = Unsigned;
    name = MinCache, id = 0x6DE7, variant = Unsigned;
    name = MaxCache, id = 0x6DF8, variant = Unsigned;
    name = DefaultDuration, id = 0x23E383, variant = Unsigned;
    name = DefaultDecodedFieldDuration, id = 0x234E7A, variant = Unsigned;
    name = TrackTimestampScale, id = 0x23314F, variant = Float;
    name = TrackOffset, id = 0x537F, variant = Signed;
    name = MaxBlockAdditionId, id = 0x55EE, variant = Unsigned;
    name = BlockAdditionMapping, id = 0x41E4, variant = Master;
    name = BlockAddIdValue, id = 0x41F0, variant = Unsigned;
    name = BlockAddIdName, id = 0x41A4, variant = String;
    name = BlockAddIdType, id = 0x41E7, variant = Unsigned;
    name = BlockAddIdExtraData, id = 0x41ED, variant = Binary;
    name = Name, id = 0x536E, variant = Utf8;
    name = Language, id = 0x22B59C, variant = String;
    name = LanguageBcp47, id = 0x22B59D, variant = String;
    name = CodecId, id = 0x86, variant = String;
    name = CodecPrivate, id = 0x63A2, variant = Binary;
    name = CodecName, id = 0x258688, variant = Utf8;
    name = AttachmentLink, id = 0x7446, variant = Unsigned;
    name = CodecSettings, id = 0x3A9697, variant = Utf8;
    name = CodecInfoUrl, id = 0x3B4040, variant = String;
    name = CodecDownloadUrl, id = 0x26B240, variant = String;
    name = CodecDecodeAll, id = 0xAA, variant = Unsigned;
    name = TrackOverlay, id = 0x6FAB, variant = Unsigned;
    name = CodecDelay, id = 0x56AA, variant = Unsigned;
    name = SeekPreRoll, id = 0x56BB, variant = Unsigned;
    name = TrackTranslate, id = 0x6624, variant = Master;
    name = TrackTranslateTrackId, id = 0x66A5, variant = Binary;
    name = TrackTranslateCodec, id = 0x66BF, variant = Unsigned;
    name = TrackTranslateEditionUid, id = 0x66FC, variant = Unsigned;
    name = Video, id = 0xE0, variant = Master;
    name = FlagInterlaced, id = 0x9A, variant = Unsigned;
    name = FieldOrder, id = 0x9D, variant = Unsigned;
    name = StereoMode, id = 0x53B8, variant = Unsigned;
    name = AlphaMode, id = 0x53C0, variant = Unsigned;
    name = OldStereoMode, id = 0x53B9, variant = Unsigned;
    name = PixelWidth, id = 0xB0, variant = Unsigned;
    name = PixelHeight, id = 0xBA, variant = Unsigned;
    name = PixelCropBottom, id = 0x54AA, variant = Unsigned;
    name = PixelCropTop, id = 0x54BB, variant = Unsigned;
    name = PixelCropLeft, id = 0x54CC, variant = Unsigned;
    name = PixelCropRight, id = 0x54DD, variant = Unsigned;
    name = DisplayWidth, id = 0x54B0, variant = Unsigned;
    name = DisplayHeight, id = 0x54BA, variant = Unsigned;
    name = DisplayUnit, id = 0x54B2, variant = Unsigned;
    name = AspectRatioType, id = 0x54B3, variant = Unsigned;
    name = UncompressedFourCc, id = 0x2EB524, variant = Binary;
    name = GammaValue, id = 0x2FB523, variant = Float;
    name = FrameRate, id = 0x2383E3, variant = Float;
    name = Colour, id = 0x55B0, variant = Master;
    name = MatrixCoefficients, id = 0x55B1, variant = Unsigned;
    name = BitsPerChannel, id = 0x55B2, variant = Unsigned;
    name = ChromaSubsamplingHorz, id = 0x55B3, variant = Unsigned;
    name = ChromaSubsamplingVert, id = 0x55B4, variant = Unsigned;
    name = CbSubsamplingHorz, id = 0x55B5, variant = Unsigned;
    name = CbSubsamplingVert, id = 0x55B6, variant = Unsigned;
    name = ChromaSitingHorz, id = 0x55B7, variant = Unsigned;
    name = ChromaSitingVert, id = 0x55B8, variant = Unsigned;
    name = Range, id = 0x55B9, variant = Unsigned;
    name = TransferCharacteristics, id = 0x55BA, variant = Unsigned;
    name = Primaries, id = 0x55BB, variant = Unsigned;
    name = MaxCll, id = 0x55BC, variant = Unsigned;
    name = MaxFall, id = 0x55BD, variant = Unsigned;
    name = MasteringMetadata, id = 0x55D0, variant = Master;
    name = PrimaryRChromaticityX, id = 0x55D1, variant = Float;
    name = PrimaryRChromaticityY, id = 0x55D2, variant = Float;
    name = PrimaryGChromaticityX, id = 0x55D3, variant = Float;
    name = PrimaryGChromaticityY, id = 0x55D4, variant = Float;
    name = PrimaryBChromaticityX, id = 0x55D5, variant = Float;
    name = PrimaryBChromaticityY, id = 0x55D6, variant = Float;
    name = WhitePointChromaticityX, id = 0x55D7, variant = Float;
    name = WhitePointChromaticityY, id = 0x55D8, variant = Float;
    name = LuminanceMax, id = 0x55D9, variant = Float;
    name = LuminanceMin, id = 0x55DA, variant = Float;
    name = Projection, id = 0x7670, variant = Master;
    name = ProjectionType, id = 0x7671, variant = Unsigned;
    name = ProjectionPrivate, id = 0x7672, variant = Binary;
    name = ProjectionPoseYaw, id = 0x7673, variant = Float;
    name = ProjectionPosePitch, id = 0x7674, variant = Float;
    name = ProjectionPoseRoll, id = 0x7675, variant = Float;
    name = Audio, id = 0xE1, variant = Master;
    name = SamplingFrequency, id = 0xB5, variant = Float;
    name = OutputSamplingFrequency, id = 0x78B5, variant = Float;
    name = Channels, id = 0x9F, variant = Unsigned;
    name = ChannelPositions, id = 0x7D7B, variant = Binary;
    name = BitDepth, id = 0x6264, variant = Unsigned;
    name = TrackOperation, id = 0xE2, variant = Master;
    name = TrackCombinePlanes, id = 0xE3, variant = Master;
    name = TrackPlane, id = 0xE4, variant = Master;
    name = TrackPlaneUid, id = 0xE5, variant = Unsigned;
    name = TrackPlaneType, id = 0xE6, variant = Unsigned;
    name = TrackJoinBlocks, id = 0xE9, variant = Master;
    name = TrackJoinUid, id = 0xED, variant = Unsigned;
    name = TrickTrackUid, id = 0xC0, variant = Unsigned;
    name = TrickTrackSegmentUid, id = 0xC1, variant = Binary;
    name = TrickTrackFlag, id = 0xC6, variant = Unsigned;
    name = TrickMasterTrackUid, id = 0xC7, variant = Unsigned;
    name = TrickMasterTrackSegmentUid, id = 0xC4, variant = Binary;
    name = ContentEncodings, id = 0x6D80, variant = Master;
    name = ContentEncoding, id = 0x6240, variant = Master;
    name = ContentEncodingOrder, id = 0x5031, variant = Unsigned;
    name = ContentEncodingScope, id = 0x5032, variant = Unsigned;
    name = ContentEncodingType, id = 0x5033, variant = Unsigned;
    name = ContentCompression, id = 0x5034, variant = Master;
    name = ContentCompAlgo, id = 0x4254, variant = Unsigned;
    name = ContentCompSettings, id = 0x4255, variant = Binary;
    name = ContentEncryption, id = 0x5035, variant = Master;
    name = ContentEncAlgo, id = 0x47E1, variant = Unsigned;
    name = ContentEncKeyId, id = 0x47E2, variant = Binary;
    name = ContentEncAesSettings, id = 0x47E7, variant = Master;
    name = AesSettingsCipherMode, id = 0x47E8, variant = Unsigned;
    name = ContentSignature, id = 0x47E3, variant = Binary;
    name = ContentSigKeyId, id = 0x47E4, variant = Binary;
    name = ContentSigAlgo, id = 0x47E5, variant = Unsigned;
    name = ContentSigHashAlgo, id = 0x47E6, variant = Unsigned;
    name = Cues, id = 0x1C53BB6B, variant = Master;
    name = CuePoint, id = 0xBB, variant = Master;
    name = CueTime, id = 0xB3, variant = Unsigned;
    name = CueTrackPositions, id = 0xB7, variant = Master;
    name = CueTrack, id = 0xF7, variant = Unsigned;
    name = CueClusterPosition, id = 0xF1, variant = Unsigned;
    name = CueRelativePosition, id = 0xF0, variant = Unsigned;
    name = CueDuration, id = 0xB2, variant = Unsigned;
    name = CueBlockNumber, id = 0x5378, variant = Unsigned;
    name = CueCodecState, id = 0xEA, variant = Unsigned;
    name = CueReference, id = 0xDB, variant = Master;
    name = CueRefTime, id = 0x96, variant = Unsigned;
    name = CueRefCluster, id = 0x97, variant = Unsigned;
    name = CueRefNumber, id = 0x535F, variant = Unsigned;
    name = CueRefCodecState, id = 0xEB, variant = Unsigned;
    name = Attachments, id = 0x1941A469, variant = Master;
    name = AttachedFile, id = 0x61A7, variant = Master;
    name = FileDescription, id = 0x467E, variant = Utf8;
    name = FileName, id = 0x466E, variant = Utf8;
    name = FileMimeType, id = 0x4660, variant = String;
    name = FileData, id = 0x465C, variant = Binary;
    name = FileUid, id = 0x46AE, variant = Unsigned;
    name = FileReferral, id = 0x4675, variant = Binary;
    name = FileUsedStartTime, id = 0x4661, variant = Unsigned;
    name = FileUsedEndTime, id = 0x4662, variant = Unsigned;
    name = Chapters, id = 0x1043A770, variant = Master;
    name = EditionEntry, id = 0x45B9, variant = Master;
    name = EditionUid, id = 0x45BC, variant = Unsigned;
    name = EditionFlagHidden, id = 0x45BD, variant = Unsigned;
    name = EditionFlagDefault, id = 0x45DB, variant = Unsigned;
    name = EditionFlagOrdered, id = 0x45DD, variant = Unsigned;
    name = ChapterAtom, id = 0xB6, variant = Master;
    name = ChapterUid, id = 0x73C4, variant = Unsigned;
    name = ChapterStringUid, id = 0x5654, variant = Utf8;
    name = ChapterTimeStart, id = 0x91, variant = Unsigned;
    name = ChapterTimeEnd, id = 0x92, variant = Unsigned;
    name = ChapterFlagHidden, id = 0x98, variant = Unsigned;
    name = ChapterFlagEnabled, id = 0x4598, variant = Unsigned;
    name = ChapterSegmentUuid, id = 0x6E67, variant = Binary;
    name = ChapterSegmentEditionUid, id = 0x6EBC, variant = Unsigned;
    name = ChapterPhysicalEquiv, id = 0x63C3, variant = Unsigned;
    name = ChapterTrack, id = 0x8F, variant = Master;
    name = ChapterTrackUid, id = 0x89, variant = Unsigned;
    name = ChapterDisplay, id = 0x80, variant = Master;
    name = ChapString, id = 0x85, variant = Utf8;
    name = ChapLanguage, id = 0x437C, variant = String;
    name = ChapLanguageBcp47, id = 0x437D, variant = String;
    name = ChapCountry, id = 0x437E, variant = String;
    name = ChapProcess, id = 0x6944, variant = Master;
    name = ChapProcessCodecId, id = 0x6955, variant = Unsigned;
    name = ChapProcessPrivate, id = 0x450D, variant = Binary;
    name = ChapProcessCommand, id = 0x6911, variant = Master;
    name = ChapProcessTime, id = 0x6922, variant = Unsigned;
    name = ChapProcessData, id = 0x6933, variant = Binary;
    name = Tags, id = 0x1254C367, variant = Master;
    name = Tag, id = 0x7373, variant = Master;
    name = Targets, id = 0x63C0, variant = Master;
    name = TargetTypeValue, id = 0x68CA, variant = Unsigned;
    name = TargetType, id = 0x63CA, variant = String;
    name = TagTrackUid, id = 0x63C5, variant = Unsigned;
    name = TagEditionUid, id = 0x63C9, variant = Unsigned;
    name = TagChapterUid, id = 0x63C4, variant = Unsigned;
    name = TagAttachmentUid, id = 0x63C6, variant = Unsigned;
    name = SimpleTag, id = 0x67C8, variant = Master;
    name = TagName, id = 0x45A3, variant = Utf8;
    name = TagLanguage, id = 0x447A, variant = String;
    name = TagLanguageBcp47, id = 0x447B, variant = String;
    name = TagDefault, id = 0x4484, variant = Unsigned;
    name = TagDefaultBogus, id = 0x44B4, variant = Unsigned;
    name = TagString, id = 0x4487, variant = Utf8;
    name = TagBinary, id = 0x4485, variant = Binary;
}
