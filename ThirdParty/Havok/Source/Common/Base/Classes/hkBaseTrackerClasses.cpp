/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#include <Common/Base/hkBase.h>
static const char s_libraryName[] = "hkBase";
#include <Common/Base/Memory/Tracker/hkTrackerClassDefinition.h>

void HK_CALL hkBaseRegister() {}

#include <Common/Base/Algorithm/Collide/1AxisSweep/hk1AxisSweep.h>


// hkKeyPair ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkKeyPair, s_libraryName)


// hk1AxisSweep ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hk1AxisSweep)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(AabbInt)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(IteratorAA)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(IteratorAB)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hk1AxisSweep, s_libraryName)


// AabbInt hk1AxisSweep
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hk1AxisSweep, AabbInt, s_libraryName)


// IteratorAA hk1AxisSweep
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hk1AxisSweep, IteratorAA, s_libraryName)


// IteratorAB hk1AxisSweep

HK_TRACKER_DECLARE_CLASS_BEGIN(hk1AxisSweep::IteratorAB)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hk1AxisSweep::IteratorAB)
    HK_TRACKER_MEMBER(hk1AxisSweep::IteratorAB, m_currentPtr, 0, "restrict hk1AxisSweep::AabbInt*") // restrict const struct hk1AxisSweep::AabbInt*
    HK_TRACKER_MEMBER(hk1AxisSweep::IteratorAB, m_potentialPtr, 0, "restrict hk1AxisSweep::AabbInt*") // restrict const struct hk1AxisSweep::AabbInt*
    HK_TRACKER_MEMBER(hk1AxisSweep::IteratorAB, m_pa, 0, "restrict hk1AxisSweep::AabbInt*") // restrict const struct hk1AxisSweep::AabbInt*
    HK_TRACKER_MEMBER(hk1AxisSweep::IteratorAB, m_pb, 0, "restrict hk1AxisSweep::AabbInt*") // restrict const struct hk1AxisSweep::AabbInt*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hk1AxisSweep::IteratorAB, s_libraryName)

#include <Common/Base/Algorithm/Collide/LineSegment/hkLineSegmentUtil.h>


// hkLineSegmentUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkLineSegmentUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ClosestLineSegLineSegResult)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ClosestPointLineSegResult)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ClosestPointInfLineInfLineResult)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(IntersectionInfLinePlaneResult)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkLineSegmentUtil, s_libraryName)


// ClosestLineSegLineSegResult hkLineSegmentUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkLineSegmentUtil, ClosestLineSegLineSegResult, s_libraryName)


// ClosestPointLineSegResult hkLineSegmentUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkLineSegmentUtil, ClosestPointLineSegResult, s_libraryName)


// ClosestPointInfLineInfLineResult hkLineSegmentUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkLineSegmentUtil, ClosestPointInfLineInfLineResult, s_libraryName)


// IntersectionInfLinePlaneResult hkLineSegmentUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkLineSegmentUtil, IntersectionInfLinePlaneResult, s_libraryName)

#include <Common/Base/Algorithm/Compression/hkBufferCompressionInternal.h>


// CompressedOutput hkBufferCompression

HK_TRACKER_DECLARE_CLASS_BEGIN(hkBufferCompression::CompressedOutput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkBufferCompression::CompressedOutput)
    HK_TRACKER_MEMBER(hkBufferCompression::CompressedOutput, m_out, 0, "hkUint8*") // hkUint8*
    HK_TRACKER_MEMBER(hkBufferCompression::CompressedOutput, m_out_start, 0, "hkUint8*") // hkUint8*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkBufferCompression::CompressedOutput, s_libraryName)

#include <Common/Base/Algorithm/Compression/hkCompression.h>

// hkCompression Result_
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkCompression::Result_, s_libraryName, hkCompression_Result_)
#include <Common/Base/Algorithm/Noise/hkPerlinNoise2d.h>


// hkPerlinNoise2d ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkPerlinNoise2d)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkPerlinNoise2d)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkPerlinNoise2d, s_libraryName, hkReferencedObject)

#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>


// hkPseudoRandomGenerator ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkPseudoRandomGenerator, s_libraryName)

#include <Common/Base/Algorithm/Sort/hkRadixSort.h>

// hk.MemoryTracker ignore hkRadixSort
#include <Common/Base/Algorithm/Sort/hkSort.h>


// ListElement hkAlgorithm

HK_TRACKER_DECLARE_CLASS_BEGIN(hkAlgorithm::ListElement)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkAlgorithm::ListElement)
    HK_TRACKER_MEMBER(hkAlgorithm::ListElement, next, 0, "hkAlgorithm::ListElement*") // struct hkAlgorithm::ListElement*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkAlgorithm::ListElement, s_libraryName)

#include <Common/Base/Algorithm/UnionFind/hkUnionFind.h>


// hkUnionFind ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkUnionFind, s_libraryName)

#include <Common/Base/Config/hkOptionalComponent.h>


// hkOptionalComponent ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkOptionalComponent)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkOptionalComponent)
    HK_TRACKER_MEMBER(hkOptionalComponent, m_next, 0, "hkOptionalComponent*") // class hkOptionalComponent*
    HK_TRACKER_MEMBER(hkOptionalComponent, m_name, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkOptionalComponent, m_funcPtr, 0, "void**") // void**
    HK_TRACKER_MEMBER(hkOptionalComponent, m_func, 0, "void*") // void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkOptionalComponent, s_libraryName)

#include <Common/Base/Config/hkProductFeatures.h>


// hkProductFeatures ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkProductFeatures, s_libraryName)

#include <Common/Base/Container/Array/hkArrayUtil.h>


// hkArrayUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkArrayUtil, s_libraryName)

#include <Common/Base/Container/BitField/hkBitField.h>


// hkBitFieldLoop ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkBitFieldLoop, s_libraryName)


// hkBitFieldBit ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkBitFieldBit, s_libraryName)


// hkBitFieldValue ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkBitFieldValue, s_libraryName)


// hkBitField ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkBitField, s_libraryName)


// hkOffsetBitField ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkOffsetBitField, s_libraryName)


// hkOffsetBitFieldLocal ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkOffsetBitFieldLocal, s_libraryName)

// hk.MemoryTracker ignore hkBitFieldStoragehkArrayunsignedinthkContainerHeapAllocator
// hk.MemoryTracker ignore hkBitFieldBasehkBitFieldStoragehkArrayunsignedinthkContainerHeapAllocator
// hk.MemoryTracker ignore hkOffsetBitFieldStoragehkArrayunsignedinthkContainerHeapAllocator
// hk.MemoryTracker ignore hkBitFieldBasehkOffsetBitFieldStoragehkArrayunsignedinthkContainerHeapAllocator
#include <Common/Base/Container/BlockStream/Allocator/Dynamic/hkDynamicBlockStreamAllocator.h>


// hkDynamicBlockStreamAllocator ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDynamicBlockStreamAllocator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDynamicBlockStreamAllocator)
    HK_TRACKER_MEMBER(hkDynamicBlockStreamAllocator, m_blocks, 0, "hkArray<hkBlockStreamBase::Block*, hkContainerHeapAllocator>") // hkArray< struct hkBlockStreamBase::Block*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkDynamicBlockStreamAllocator, m_freeList, 0, "hkArray<hkBlockStreamBase::Block*, hkContainerHeapAllocator>") // hkArray< struct hkBlockStreamBase::Block*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkDynamicBlockStreamAllocator, s_libraryName, hkBlockStreamAllocator)

#include <Common/Base/Container/BlockStream/Allocator/Fixed/hkFixedBlockStreamAllocator.h>


// hkFixedBlockStreamAllocator ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkFixedBlockStreamAllocator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkFixedBlockStreamAllocator)
    HK_TRACKER_MEMBER(hkFixedBlockStreamAllocator, m_blocks, 0, "hkBlockStreamBase::Block*") // struct hkBlockStreamBase::Block*
    HK_TRACKER_MEMBER(hkFixedBlockStreamAllocator, m_freeList, 0, "hkArray<hkBlockStreamBase::Block*, hkContainerHeapAllocator>") // hkArray< struct hkBlockStreamBase::Block*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkFixedBlockStreamAllocator, s_libraryName, hkBlockStreamAllocator)

#include <Common/Base/Container/BlockStream/Allocator/hkBlockStreamAllocator.h>


// hkBlockStreamAllocator ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkBlockStreamAllocator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkBlockStreamAllocator)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkBlockStreamAllocator, s_libraryName, hkReferencedObject)

#include <Common/Base/Container/BlockStream/Allocator/hkThreadLocalBlockStreamAllocator.h>


// hkThreadLocalBlockStreamAllocator ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkThreadLocalBlockStreamAllocator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkThreadLocalBlockStreamAllocator)
    HK_TRACKER_MEMBER(hkThreadLocalBlockStreamAllocator, m_blockStreamAllocator, 0, "hkPadSpu<hkBlockStreamAllocator*>") // class hkPadSpu< class hkBlockStreamAllocator* >
    HK_TRACKER_MEMBER(hkThreadLocalBlockStreamAllocator, m_freeBlocks, 0, "hkBlockStreamBase::Block* [8]") // struct hkBlockStreamBase::Block* [8]
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkThreadLocalBlockStreamAllocator, s_libraryName)

#include <Common/Base/Container/BlockStream/hkBlockStreamBaseBlock.h>


// Block hkBlockStreamBase

HK_TRACKER_DECLARE_CLASS_BEGIN(hkBlockStreamBase::Block)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkBlockStreamBase::Block)
    HK_TRACKER_MEMBER(hkBlockStreamBase::Block, m_next, 0, "hkBlockStreamBase::Block*") // struct hkBlockStreamBase::Block*
    HK_TRACKER_MEMBER(hkBlockStreamBase::Block, m_allocator, 0, "hkBlockStreamAllocator*") // class hkBlockStreamAllocator*
    HK_TRACKER_MEMBER(hkBlockStreamBase::Block, m_blockStream, 0, "hkBlockStreamBase::Stream*") // class hkBlockStreamBase::Stream*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkBlockStreamBase::Block, s_libraryName)

#include <Common/Base/Container/CommandStream/hkCommandStream.h>


// hkCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkCommand, s_libraryName)


// hkSecondaryCommandDispatcher ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkSecondaryCommandDispatcher)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkSecondaryCommandDispatcher)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkSecondaryCommandDispatcher, s_libraryName, hkReferencedObject)


// hkBlockStreamCommandWriter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkBlockStreamCommandWriter)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkBlockStreamCommandWriter)
    HK_TRACKER_MEMBER(hkBlockStreamCommandWriter, m_writer, 0, "hkBlockStream<hkCommand>::Writer") // class hkBlockStream<hkCommand>::Writer
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkBlockStreamCommandWriter, s_libraryName, hkSecondaryCommandDispatcher)


// hkPrimaryCommandDispatcher ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkPrimaryCommandDispatcher)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkPrimaryCommandDispatcher)
    HK_TRACKER_MEMBER(hkPrimaryCommandDispatcher, m_commandDispatcher, 0, "hkSecondaryCommandDispatcher * [7]") // class hkSecondaryCommandDispatcher * [7]
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkPrimaryCommandDispatcher, s_libraryName)

#include <Common/Base/Container/ObjectPackUtility/hkObjectPackUtility.h>


// hkObjectUnpackUtility ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkObjectUnpackUtility)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkObjectUnpackUtility)
    HK_TRACKER_MEMBER(hkObjectUnpackUtility, m_newObject, 0, "void*") // void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkObjectUnpackUtility, s_libraryName)

#include <Common/Base/Container/PointerMap/hkMap.h>

 // Skipping Class hkMapOperations< int > as it is a template

#include <Common/Base/Container/PointerMap/hkPointerMap.h>

 // Skipping Class hkPointerMapStorage< 8 > as it is a template

#include <Common/Base/Container/PointerMultiMap/hkMultiMap.h>


// hkMultiMapIntegralOperations ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkMultiMapIntegralOperations, s_libraryName)

#include <Common/Base/Container/PointerMultiMap/hkPointerMultiMap.h>

 // Skipping Class hkPointerMultiMapStorage< 8 > as it is a template

 // Skipping Class hkPointerMultiMapSelectOperations< int > as it is a template

 // Skipping Class hkPointerMultiMapSelectOperations< hkUint32 > as it is a template

#include <Common/Base/Container/RelArray/hkRelArrayUtil.h>


// hkRelArrayUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkRelArrayUtil, s_libraryName)

#include <Common/Base/Container/Set/hkSet.h>


// hkSetUint32 ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkSetUint32, s_libraryName)


// hkIntRealPair ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkIntRealPair, s_libraryName)

 // Skipping Class hkMapOperations< struct hkIntRealPair > as it is a template


// hkSetIntFloatPair ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkSetIntFloatPair, s_libraryName)

// hk.MemoryTracker ignore hkSetunsignedinthkContainerHeapAllocatorhkMapOperationsunsignedint
// hk.MemoryTracker ignore hkSethkIntRealPairhkContainerHeapAllocatorhkMapOperationshkIntRealPair
#include <Common/Base/Container/String/Deprecated/hkStringOld.h>


// hkStringOld ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkStringOld)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkStringOld)
    HK_TRACKER_MEMBER(hkStringOld, m_string, 0, "hkArray<char, hkContainerHeapAllocator>") // hkArray< char, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkStringOld, s_libraryName)

#include <Common/Base/Container/String/hkString.h>

// hkString ReplaceType
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkString::ReplaceType, s_libraryName, hkString_ReplaceType)
#include <Common/Base/Container/String/hkStringBuf.h>


// hkStringBuf ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkStringBuf)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ReplaceType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkStringBuf)
    HK_TRACKER_MEMBER(hkStringBuf, m_string, 0, "hkInplaceArray<char, 128, hkContainerTempAllocator>") // class hkInplaceArray< char, 128, struct hkContainerTempAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkStringBuf, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkStringBuf, ReplaceType, s_libraryName)

#include <Common/Base/Container/String/hkStringObject.h>


// hkStringObject ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkStringObject)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkStringObject)
    HK_TRACKER_MEMBER(hkStringObject, m_string, 0, "hkStringPtr") // hkStringPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkStringObject, s_libraryName, hkReferencedObject)

#include <Common/Base/Container/String/hkStringPtr.h>


// hkStringPtr ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkStringPtr)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(StringFlags)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkStringPtr)
    HK_TRACKER_MEMBER(hkStringPtr, m_stringAndFlag, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkStringPtr, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkStringPtr, StringFlags, s_libraryName)

#include <Common/Base/Container/String/hkUtf8.h>


// Utf8FromWide hkUtf8

HK_TRACKER_DECLARE_CLASS_BEGIN(hkUtf8::Utf8FromWide)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkUtf8::Utf8FromWide)
    HK_TRACKER_MEMBER(hkUtf8::Utf8FromWide, m_utf8, 0, "hkArray<char, hkContainerTempAllocator>") // hkArray< char, struct hkContainerTempAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkUtf8::Utf8FromWide, s_libraryName)


// WideFromUtf8 hkUtf8

HK_TRACKER_DECLARE_CLASS_BEGIN(hkUtf8::WideFromUtf8)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkUtf8::WideFromUtf8)
    HK_TRACKER_MEMBER(hkUtf8::WideFromUtf8, m_wide, 0, "hkArray<wchar_t, hkContainerTempAllocator>") // hkArray< wchar_t, struct hkContainerTempAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkUtf8::WideFromUtf8, s_libraryName)


// Iterator hkUtf8

HK_TRACKER_DECLARE_CLASS_BEGIN(hkUtf8::Iterator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkUtf8::Iterator)
    HK_TRACKER_MEMBER(hkUtf8::Iterator, m_utf8, 0, "hkUint8*") // const hkUint8*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkUtf8::Iterator, s_libraryName)

#include <Common/Base/Container/StringDictionary/hkStringDictionary.h>


// hkStringDictionary ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkStringDictionary)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkStringDictionary)
    HK_TRACKER_MEMBER(hkStringDictionary, m_dictionary, 0, "hkArray<char*, hkContainerHeapAllocator>") // hkArray< const char*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkStringDictionary, s_libraryName)

#include <Common/Base/Container/StringMap/hkStringMap.h>


// hkStringMapOperations ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkStringMapOperations, s_libraryName)

#include <Common/Base/Container/SubString/hkSubString.h>


// hkSubString ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkSubString)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkSubString)
    HK_TRACKER_MEMBER(hkSubString, m_start, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkSubString, m_end, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkSubString, s_libraryName)

#include <Common/Base/Container/Tree/hkAnderssonTree.h>


// hkAATree ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkAATree)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Node)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Iterator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkAATree)
    HK_TRACKER_MEMBER(hkAATree, m_root, 0, "hkAATree::Node*") // struct hkAATree::Node*
    HK_TRACKER_MEMBER(hkAATree, m_nil, 0, "hkAATree::Node*") // struct hkAATree::Node*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkAATree, s_libraryName)


// Node hkAATree

HK_TRACKER_DECLARE_CLASS_BEGIN(hkAATree::Node)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkAATree::Node)
    HK_TRACKER_MEMBER(hkAATree::Node, m_data, 0, "void*") // void*
    HK_TRACKER_MEMBER(hkAATree::Node, m_link, 0, "hkAATree::Node* [2]") // struct hkAATree::Node* [2]
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkAATree::Node, s_libraryName)


// Iterator hkAATree

HK_TRACKER_DECLARE_CLASS_BEGIN(hkAATree::Iterator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkAATree::Iterator)
    HK_TRACKER_MEMBER(hkAATree::Iterator, m_tree, 0, "hkAATree*") // class hkAATree*
    HK_TRACKER_MEMBER(hkAATree::Iterator, m_it, 0, "hkAATree::Node*") // struct hkAATree::Node*
    HK_TRACKER_MEMBER(hkAATree::Iterator, m_path, 0, "hkAATree::Node* [64]") // struct hkAATree::Node* [64]
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkAATree::Iterator, s_libraryName)

#include <Common/Base/Container/Tree/hkSortedTree.h>


// hkSortedTreeBase ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkSortedTreeBase)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CompareValues)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ComparePointers)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Prng)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkSortedTreeBase, s_libraryName)


// CompareValues hkSortedTreeBase
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkSortedTreeBase, CompareValues, s_libraryName)


// ComparePointers hkSortedTreeBase
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkSortedTreeBase, ComparePointers, s_libraryName)


// Prng hkSortedTreeBase
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkSortedTreeBase, Prng, s_libraryName)

#include <Common/Base/Container/hkContainerAllocators.h>


// hkContainerTempAllocator ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkContainerTempAllocator)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Allocator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkContainerTempAllocator, s_libraryName)


// Allocator hkContainerTempAllocator

HK_TRACKER_DECLARE_CLASS_BEGIN(hkContainerTempAllocator::Allocator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkContainerTempAllocator::Allocator)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkContainerTempAllocator::Allocator, s_libraryName, hkMemoryAllocator)


// hkContainerHeapAllocator ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkContainerHeapAllocator)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Allocator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkContainerHeapAllocator, s_libraryName)


// Allocator hkContainerHeapAllocator

HK_TRACKER_DECLARE_CLASS_BEGIN(hkContainerHeapAllocator::Allocator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkContainerHeapAllocator::Allocator)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkContainerHeapAllocator::Allocator, s_libraryName, hkMemoryAllocator)


// hkContainerDebugAllocator ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkContainerDebugAllocator)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Allocator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkContainerDebugAllocator, s_libraryName)


// Allocator hkContainerDebugAllocator

HK_TRACKER_DECLARE_CLASS_BEGIN(hkContainerDebugAllocator::Allocator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkContainerDebugAllocator::Allocator)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkContainerDebugAllocator::Allocator, s_libraryName, hkMemoryAllocator)


// hkContainerDefaultMallocAllocator ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkContainerDefaultMallocAllocator, s_libraryName)

#include <Common/Base/DebugUtil/DeterminismUtil/hkNetworkedDeterminismUtil.h>


// hkNetworkedDeterminismUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkNetworkedDeterminismUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Command)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Server)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Client)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ControlCommand)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(DeterminismDataCommand)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkNetworkedDeterminismUtil)
    HK_TRACKER_MEMBER(hkNetworkedDeterminismUtil, m_serverAddress, 0, "hkStringPtr") // hkStringPtr
    HK_TRACKER_MEMBER(hkNetworkedDeterminismUtil, m_server, 0, "hkNetworkedDeterminismUtil::Server*") // class hkNetworkedDeterminismUtil::Server*
    HK_TRACKER_MEMBER(hkNetworkedDeterminismUtil, m_client, 0, "hkNetworkedDeterminismUtil::Client*") // class hkNetworkedDeterminismUtil::Client*
    HK_TRACKER_MEMBER(hkNetworkedDeterminismUtil, m_controlCommand, 0, "hkNetworkedDeterminismUtil::ControlCommand") // struct hkNetworkedDeterminismUtil::ControlCommand
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkNetworkedDeterminismUtil, s_libraryName)


// Command hkNetworkedDeterminismUtil

HK_TRACKER_DECLARE_CLASS_BEGIN(hkNetworkedDeterminismUtil::Command)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Type)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkNetworkedDeterminismUtil::Command)
    HK_TRACKER_MEMBER(hkNetworkedDeterminismUtil::Command, m_data, 0, "hkArray<hkUint8, hkContainerHeapAllocator>") // hkArray< hkUint8, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkNetworkedDeterminismUtil::Command, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkNetworkedDeterminismUtil::Command, Type, s_libraryName)


// Server hkNetworkedDeterminismUtil

HK_TRACKER_DECLARE_CLASS_BEGIN(hkNetworkedDeterminismUtil::Server)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkNetworkedDeterminismUtil::Server)
    HK_TRACKER_MEMBER(hkNetworkedDeterminismUtil::Server, m_listeningSocket, 0, "hkSocket*") // class hkSocket*
    HK_TRACKER_MEMBER(hkNetworkedDeterminismUtil::Server, m_clients, 0, "hkArray<hkSocket*, hkContainerHeapAllocator>") // hkArray< class hkSocket*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkNetworkedDeterminismUtil::Server, s_libraryName)


// Client hkNetworkedDeterminismUtil

HK_TRACKER_DECLARE_CLASS_BEGIN(hkNetworkedDeterminismUtil::Client)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkNetworkedDeterminismUtil::Client)
    HK_TRACKER_MEMBER(hkNetworkedDeterminismUtil::Client, m_socket, 0, "hkSocket*") // class hkSocket*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkNetworkedDeterminismUtil::Client, s_libraryName)


// ControlCommand hkNetworkedDeterminismUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkNetworkedDeterminismUtil, ControlCommand, s_libraryName)


// DeterminismDataCommand hkNetworkedDeterminismUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkNetworkedDeterminismUtil, DeterminismDataCommand, s_libraryName)

#include <Common/Base/DebugUtil/GlobalProperties/hkGlobalProperties.h>


// hkGlobalProperties ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkGlobalProperties)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkGlobalProperties)
    HK_TRACKER_MEMBER(hkGlobalProperties, m_data, 0, "hkPointerMap<void*, void*, hkContainerHeapAllocator>") // class hkPointerMap< const void*, const void*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkGlobalProperties, m_lock, 0, "hkCriticalSection*") // class hkCriticalSection*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkGlobalProperties, s_libraryName, hkReferencedObject)

#include <Common/Base/DebugUtil/Logger/hkLogger.h>


// hkLogger ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkLogger)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(LogLevel)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkLogger)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkLogger, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkLogger, LogLevel, s_libraryName)

#include <Common/Base/DebugUtil/Logger/hkOstreamLogger.h>


// hkOstreamLogger ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkOstreamLogger)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkOstreamLogger)
    HK_TRACKER_MEMBER(hkOstreamLogger, m_os, 0, "hkOstream") // class hkOstream
    HK_TRACKER_MEMBER(hkOstreamLogger, m_indent, 0, "hkArray<char, hkContainerHeapAllocator>") // hkArray< char, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkOstreamLogger, m_scopes, 0, "hkArray<char*, hkContainerHeapAllocator>") // hkArray< const char*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkOstreamLogger, s_libraryName, hkLogger)

#include <Common/Base/DebugUtil/MemoryExceptionTestingUtil/hkMemoryExceptionTestingUtil.h>


// hkMemoryExceptionTestingUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkMemoryExceptionTestingUtil, s_libraryName)

#include <Common/Base/DebugUtil/MultiThreadCheck/hkMultiThreadCheck.h>


// hkMultiThreadCheck ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkMultiThreadCheck, s_libraryName)

#include <Common/Base/DebugUtil/TraceStream/hkTraceStream.h>


// hkTraceStream ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkTraceStream)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Title)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkTraceStream)
    HK_TRACKER_MEMBER(hkTraceStream, m_stream, 0, "hkOstream*") // class hkOstream*
    HK_TRACKER_MEMBER(hkTraceStream, m_titles, 0, "hkArray<hkTraceStream::Title, hkContainerHeapAllocator>") // hkArray< struct hkTraceStream::Title, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkTraceStream, s_libraryName, hkReferencedObject)


// Title hkTraceStream
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkTraceStream, Title, s_libraryName)

#include <Common/Base/Math/ExtendedMath/hkMpMath.h>


// hkMpUint ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMpUint)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMpUint)
    HK_TRACKER_MEMBER(hkMpUint, m_atoms, 0, "hkInplaceArray<hkUint32, 8, hkContainerHeapAllocator>") // class hkInplaceArray< hkUint32, 8, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkMpUint, s_libraryName)


// hkMpRational ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMpRational)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMpRational)
    HK_TRACKER_MEMBER(hkMpRational, m_num, 0, "hkMpUint") // struct hkMpUint
    HK_TRACKER_MEMBER(hkMpRational, m_den, 0, "hkMpUint") // struct hkMpUint
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkMpRational, s_libraryName)

#include <Common/Base/Math/Float16Transform/hkFloat16Transform.h>


// hkFloat16Transform ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkFloat16Transform, s_libraryName)

#include <Common/Base/Math/Functions/hkMathFuncs.h>

 // Skipping Class Log2< 1 > as it is a template

#include <Common/Base/Math/Header/hkMathHeaderConstantDefinitions.h>

// None hkVectorConstant
HK_TRACKER_IMPLEMENT_SIMPLE(hkVectorConstant, s_libraryName)
// None hkIntVectorConstant
HK_TRACKER_IMPLEMENT_SIMPLE(hkIntVectorConstant, s_libraryName)
#include <Common/Base/Math/Header/hkMathHeaderEnums.h>


// hkVectorPermutation ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkVectorPermutation, s_libraryName)


// hkVector4ComparisonMask ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkVector4ComparisonMask, s_libraryName)


// hkMxVectorPermutation ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkMxVectorPermutation, s_libraryName)

// None hkMathAccuracyMode
HK_TRACKER_IMPLEMENT_SIMPLE(hkMathAccuracyMode, s_libraryName)
// None hkMathDivByZeroMode
HK_TRACKER_IMPLEMENT_SIMPLE(hkMathDivByZeroMode, s_libraryName)
// None hkMathNegSqrtMode
HK_TRACKER_IMPLEMENT_SIMPLE(hkMathNegSqrtMode, s_libraryName)
// None hkMathIoMode
HK_TRACKER_IMPLEMENT_SIMPLE(hkMathIoMode, s_libraryName)
// None hkMathRoundingMode
HK_TRACKER_IMPLEMENT_SIMPLE(hkMathRoundingMode, s_libraryName)
// None hkMathValueType
HK_TRACKER_IMPLEMENT_SIMPLE(hkMathValueType, s_libraryName)
// None hkMathSortDir
HK_TRACKER_IMPLEMENT_SIMPLE(hkMathSortDir, s_libraryName)
#include <Common/Base/Math/Header/hkMathHeaderForwardDeclarations.h>

 // Skipping Class hkRealTypes< float > as it is a template

 // Skipping Class hkRealTypes< double > as it is a template

#include <Common/Base/Math/Matrix/hkMatrix3d.h>

 // Skipping Class HK_MATRIX3d_FUNC_NOT_IMPLEMENTED< 1 > as it is a template


// hkMatrix3d ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkMatrix3d, s_libraryName)

#include <Common/Base/Math/Matrix/hkMatrix3f.h>

 // Skipping Class HK_MATRIX3f_FUNC_NOT_IMPLEMENTED< 1 > as it is a template


// hkMatrix3f ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkMatrix3f, s_libraryName)

#include <Common/Base/Math/Matrix/hkMatrix4d.h>

 // Skipping Class HK_MATRIX4d_FUNC_NOT_IMPLEMENTED< 1 > as it is a template


// hkMatrix4d ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkMatrix4d, s_libraryName)

#include <Common/Base/Math/Matrix/hkMatrix4f.h>

 // Skipping Class HK_MATRIX4f_FUNC_NOT_IMPLEMENTED< 1 > as it is a template


// hkMatrix4f ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkMatrix4f, s_libraryName)

#include <Common/Base/Math/Matrix/hkMatrix6d.h>


// hkVector8d ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkVector8d, s_libraryName)


// hkMatrix6d ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkMatrix6d, s_libraryName)

#include <Common/Base/Math/Matrix/hkMatrix6f.h>


// hkVector8f ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkVector8f, s_libraryName)


// hkMatrix6f ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkMatrix6f, s_libraryName)

#include <Common/Base/Math/Matrix/hkMatrixdNm.h>


// hkMatrixdNm ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMatrixdNm)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMatrixdNm)
    HK_TRACKER_MEMBER(hkMatrixdNm, m_elements, 0, "hkArray<hkVector4d, hkContainerHeapAllocator>") // hkArray< hkVector4d, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkMatrixdNm, s_libraryName)

#include <Common/Base/Math/Matrix/hkMatrixfNm.h>


// hkMatrixfNm ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMatrixfNm)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMatrixfNm)
    HK_TRACKER_MEMBER(hkMatrixfNm, m_elements, 0, "hkArray<hkVector4f, hkContainerHeapAllocator>") // hkArray< hkVector4f, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkMatrixfNm, s_libraryName)

#include <Common/Base/Math/Matrix/hkRotationd.h>


// hkRotationd ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkRotationd, s_libraryName)

#include <Common/Base/Math/Matrix/hkRotationf.h>


// hkRotationf ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkRotationf, s_libraryName)

#include <Common/Base/Math/Matrix/hkTransformd.h>


// hkTransformd ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkTransformd, s_libraryName)

#include <Common/Base/Math/Matrix/hkTransformf.h>


// hkTransformf ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkTransformf, s_libraryName)

#include <Common/Base/Math/QTransform/hkQTransformd.h>


// hkQTransformd ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkQTransformd, s_libraryName)

#include <Common/Base/Math/QTransform/hkQTransformf.h>


// hkQTransformf ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkQTransformf, s_libraryName)

#include <Common/Base/Math/QsTransform/hkQsTransformd.h>


// hkQsTransformd ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkQsTransformd, s_libraryName)

#include <Common/Base/Math/QsTransform/hkQsTransformf.h>


// hkQsTransformf ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkQsTransformf, s_libraryName)

#include <Common/Base/Math/Quaternion/hkQuaterniond.h>


// hkQuaterniond ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkQuaterniond, s_libraryName)

#include <Common/Base/Math/Quaternion/hkQuaternionf.h>


// hkQuaternionf ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkQuaternionf, s_libraryName)

#include <Common/Base/Math/SweptTransform/hkSweptTransformd.h>


// hkSweptTransformd ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkSweptTransformd, s_libraryName)

#include <Common/Base/Math/SweptTransform/hkSweptTransformf.h>


// hkSweptTransformf ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkSweptTransformf, s_libraryName)

#include <Common/Base/Math/Vector/hk4xVector2d.h>


// hk4xVector2d ::
HK_TRACKER_IMPLEMENT_SIMPLE(hk4xVector2d, s_libraryName)

#include <Common/Base/Math/Vector/hk4xVector2f.h>


// hk4xVector2f ::
HK_TRACKER_IMPLEMENT_SIMPLE(hk4xVector2f, s_libraryName)

#include <Common/Base/Math/Vector/hkFourTransposedPointsd.h>


// hkFourTransposedPointsd ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkFourTransposedPointsd, s_libraryName)

#include <Common/Base/Math/Vector/hkFourTransposedPointsf.h>


// hkFourTransposedPointsf ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkFourTransposedPointsf, s_libraryName)

#include <Common/Base/Math/Vector/hkIntVector.h>

 // Skipping Class HK_INT_VECTOR_SUBVECTOR_INDEX_OUT_OF_RANGE< 1 > as it is a template

 // Skipping Class HK_INT_VECTOR_NOT_IMPLEMENTED_FOR_THIS_VECTOR_LENGTH< 1 > as it is a template

 // Skipping Class HK_INT_VECTOR_UNSUPPORTED_VECTOR_LENGTH< 1 > as it is a template

 // Skipping Class HK_INT_VECTOR_ILLEGAL_VALUE_FOR_IMM_SPLAT< 1 > as it is a template


// hkIntVector ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkIntVector, s_libraryName)

#include <Common/Base/Math/Vector/hkPackedTransform.h>


// hkPackedTransform ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkPackedTransform, s_libraryName)

#include <Common/Base/Math/Vector/hkPackedVector3.h>


// hkPackedVector3 ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkPackedVector3, s_libraryName)


// hkPackedVector8_3 ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkPackedVector8_3, s_libraryName)

#include <Common/Base/Math/Vector/hkSimdDouble64.h>

 // Skipping Class HK_SIMDDOUBLE_TEMPLATE_CONFIGURATION_NOT_IMPLEMENTED< 1 > as it is a template

 // Skipping Class HK_SIMDDOUBLE_ILLEGAL_DIMENSION_USED< 1 > as it is a template

 // Skipping Class HK_SIMDDOUBLE_ILLEGAL_CONSTANT_REQUEST< 1 > as it is a template


// hkSimdDouble64 ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkSimdDouble64, s_libraryName)

#include <Common/Base/Math/Vector/hkSimdFloat32.h>

 // Skipping Class HK_SIMDFLOAT_TEMPLATE_CONFIGURATION_NOT_IMPLEMENTED< 1 > as it is a template

 // Skipping Class HK_SIMDFLOAT_ILLEGAL_DIMENSION_USED< 1 > as it is a template

 // Skipping Class HK_SIMDFLOAT_ILLEGAL_CONSTANT_REQUEST< 1 > as it is a template


// hkSimdFloat32 ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkSimdFloat32, s_libraryName)

#include <Common/Base/Math/Vector/hkVector2d.h>


// hkVector2d ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkVector2d, s_libraryName)

#include <Common/Base/Math/Vector/hkVector2f.h>


// hkVector2f ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkVector2f, s_libraryName)

#include <Common/Base/Math/Vector/hkVector4d.h>

 // Skipping Class HK_VECTOR4d_TEMPLATE_CONFIGURATION_NOT_IMPLEMENTED< 1 > as it is a template

 // Skipping Class HK_VECTOR4d_SUBVECTOR_INDEX_OUT_OF_RANGE< 1 > as it is a template

 // Skipping Class HK_VECTOR4d_NOT_IMPLEMENTED_FOR_THIS_VECTOR_LENGTH< 1 > as it is a template

 // Skipping Class HK_VECTOR4d_UNSUPPORTED_VECTOR_LENGTH< 1 > as it is a template


// hkVector4d ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkVector4d, s_libraryName)

#include <Common/Base/Math/Vector/hkVector4f.h>

 // Skipping Class HK_VECTOR4f_TEMPLATE_CONFIGURATION_NOT_IMPLEMENTED< 1 > as it is a template

 // Skipping Class HK_VECTOR4f_SUBVECTOR_INDEX_OUT_OF_RANGE< 1 > as it is a template

 // Skipping Class HK_VECTOR4f_NOT_IMPLEMENTED_FOR_THIS_VECTOR_LENGTH< 1 > as it is a template

 // Skipping Class HK_VECTOR4f_UNSUPPORTED_VECTOR_LENGTH< 1 > as it is a template


// hkVector4f ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkVector4f, s_libraryName)

#include <Common/Base/Math/Vector/hkVectorNd.h>


// hkVectorNd ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkVectorNd)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkVectorNd)
    HK_TRACKER_MEMBER(hkVectorNd, m_elements, 0, "hkVector4d*") // hkVector4d*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkVectorNd, s_libraryName)

#include <Common/Base/Math/Vector/hkVectorNf.h>


// hkVectorNf ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkVectorNf)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkVectorNf)
    HK_TRACKER_MEMBER(hkVectorNf, m_elements, 0, "hkVector4f*") // hkVector4f*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkVectorNf, s_libraryName)

#include <Common/Base/Memory/Allocator/Checking/hkDelayedFreeAllocator.h>

// hk.MemoryTracker ignore hkDelayedFreeAllocator
#include <Common/Base/Memory/Allocator/Checking/hkLeakDetectAllocator.h>

// hk.MemoryTracker ignore hkLeakDetectAllocator
#include <Common/Base/Memory/Allocator/Checking/hkPaddedAllocator.h>

// hk.MemoryTracker ignore hkPaddedAllocator
#include <Common/Base/Memory/Allocator/FreeList/hkFixedSizeAllocator.h>


// hkFixedSizeAllocator ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkFixedSizeAllocator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkFixedSizeAllocator)
    HK_TRACKER_MEMBER(hkFixedSizeAllocator, m_freeList, 0, "hkFreeList") // class hkFreeList
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkFixedSizeAllocator, s_libraryName, hkMemoryAllocator)

#include <Common/Base/Memory/Allocator/FreeList/hkFreeList.h>


// hkFreeList ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkFreeList)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Element)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Block)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkFreeList)
    HK_TRACKER_MEMBER(hkFreeList, m_free, 0, "hkFreeList::Element*") // struct hkFreeList::Element*
    HK_TRACKER_MEMBER(hkFreeList, m_lastIncrementalBlock, 0, "hkFreeList::Block*") // struct hkFreeList::Block*
    HK_TRACKER_MEMBER(hkFreeList, m_activeBlocks, 0, "hkFreeList::Block*") // struct hkFreeList::Block*
    HK_TRACKER_MEMBER(hkFreeList, m_freeBlocks, 0, "hkFreeList::Block*") // struct hkFreeList::Block*
    HK_TRACKER_MEMBER(hkFreeList, m_top, 0, "hkUint8*") // hkUint8*
    HK_TRACKER_MEMBER(hkFreeList, m_blockEnd, 0, "hkUint8*") // hkUint8*
    HK_TRACKER_MEMBER(hkFreeList, m_elementAllocator, 0, "hkMemoryAllocator*") // class hkMemoryAllocator*
    HK_TRACKER_MEMBER(hkFreeList, m_blockAllocator, 0, "hkMemoryAllocator*") // class hkMemoryAllocator*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkFreeList, s_libraryName)


// Element hkFreeList

HK_TRACKER_DECLARE_CLASS_BEGIN(hkFreeList::Element)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkFreeList::Element)
    HK_TRACKER_MEMBER(hkFreeList::Element, m_next, 0, "hkFreeList::Element*") // struct hkFreeList::Element*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkFreeList::Element, s_libraryName)


// Block hkFreeList

HK_TRACKER_DECLARE_CLASS_BEGIN(hkFreeList::Block)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkFreeList::Block)
    HK_TRACKER_MEMBER(hkFreeList::Block, m_next, 0, "hkFreeList::Block*") // struct hkFreeList::Block*
    HK_TRACKER_MEMBER(hkFreeList::Block, m_elementsAlloc, 0, "hkUint8*") // hkUint8*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkFreeList::Block, s_libraryName)

#include <Common/Base/Memory/Allocator/FreeList/hkFreeListAllocator.h>


// hkFreeListAllocator ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkFreeListAllocator)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(FreeListCinfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Cinfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkFreeListAllocator)
    HK_TRACKER_MEMBER(hkFreeListAllocator, m_allocator, 0, "hkMemoryAllocator*") // class hkMemoryAllocator*
    HK_TRACKER_MEMBER(hkFreeListAllocator, m_allocatorExtended, 0, "hkMemoryAllocator::ExtendedInterface*") // struct hkMemoryAllocator::ExtendedInterface*
    HK_TRACKER_MEMBER(hkFreeListAllocator, m_blockAllocator, 0, "hkFixedSizeAllocator") // class hkFixedSizeAllocator
    HK_TRACKER_MEMBER(hkFreeListAllocator, m_sizeToFreeList, 0, "hkFreeList* [41]") // class hkFreeList* [41]
    HK_TRACKER_MEMBER(hkFreeListAllocator, m_freeLists, 0, "hkFreeList* [41]") // class hkFreeList* [41]
    HK_TRACKER_MEMBER(hkFreeListAllocator, m_topFreeList, 0, "hkFreeList*") // class hkFreeList*
    HK_TRACKER_MEMBER(hkFreeListAllocator, m_lastFreeList, 0, "hkFreeList*") // class hkFreeList*
    HK_TRACKER_MEMBER(hkFreeListAllocator, m_freeListMemory, 0, "hkFreeList [41]") // class hkFreeList [41]
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkFreeListAllocator, s_libraryName, hkMemoryAllocator)


// FreeListCinfo hkFreeListAllocator
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkFreeListAllocator, FreeListCinfo, s_libraryName)


// Cinfo hkFreeListAllocator
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkFreeListAllocator, Cinfo, s_libraryName)

#include <Common/Base/Memory/Allocator/LargeBlock/hkLargeBlockAllocator.h>

// hk.MemoryTracker ignore hkLargeBlockAllocator
#include <Common/Base/Memory/Allocator/Lifo/hkLifoAllocator.h>

// hk.MemoryTracker ignore hkLifoAllocator
#include <Common/Base/Memory/Allocator/Linear/hkLinearAllocator.h>

// hk.MemoryTracker ignore hkLinearAllocator
#include <Common/Base/Memory/Allocator/Malloc/hkMallocAllocator.h>

// hk.MemoryTracker ignore hkMallocAllocator
#include <Common/Base/Memory/Allocator/Monitor/hkMonitorAllocator.h>


// hkMonitorAllocator ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMonitorAllocator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMonitorAllocator)
    HK_TRACKER_MEMBER(hkMonitorAllocator, m_child, 0, "hkMemoryAllocator*") // class hkMemoryAllocator*
    HK_TRACKER_MEMBER(hkMonitorAllocator, m_monitorTag, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkMonitorAllocator, s_libraryName, hkMemoryAllocator)

#include <Common/Base/Memory/Allocator/Pooled/hkPooledAllocator.h>

// hk.MemoryTracker ignore hkPooledAllocator
#include <Common/Base/Memory/Allocator/Recall/hkRecallAllocator.h>

// hk.MemoryTracker ignore hkRecallAllocator
#include <Common/Base/Memory/Allocator/Solver/hkSolverAllocator.h>


// hkSolverAllocator ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkSolverAllocator)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Element)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkSolverAllocator)
    HK_TRACKER_MEMBER(hkSolverAllocator, m_bufferStart, 0, "char*") // char*
    HK_TRACKER_MEMBER(hkSolverAllocator, m_bufferEnd, 0, "char*") // char*
    HK_TRACKER_MEMBER(hkSolverAllocator, m_currentEnd, 0, "char*") // char*
    HK_TRACKER_MEMBER(hkSolverAllocator, m_freeElems, 0, "hkArrayBase<hkSolverAllocator::Element>") // class hkArrayBase< struct hkSolverAllocator::Element >
    HK_TRACKER_MEMBER(hkSolverAllocator, m_elemsBuf, 0, "hkSolverAllocator::Element [64]") // struct hkSolverAllocator::Element [64]
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkSolverAllocator, s_libraryName, hkMemoryAllocator)


// Element hkSolverAllocator

HK_TRACKER_DECLARE_CLASS_BEGIN(hkSolverAllocator::Element)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkSolverAllocator::Element)
    HK_TRACKER_MEMBER(hkSolverAllocator::Element, m_start, 0, "char*") // char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkSolverAllocator::Element, s_libraryName)

#include <Common/Base/Memory/Allocator/Stats/hkStatsAllocator.h>

// hk.MemoryTracker ignore hkStatsAllocator
#include <Common/Base/Memory/Allocator/TempDetect/hkTempDetectAllocator.h>


// hkTempDetectAllocator ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkTempDetectAllocator)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(AllocInfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SizeInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkTempDetectAllocator)
    HK_TRACKER_MEMBER(hkTempDetectAllocator, m_child, 0, "hkMemoryAllocator*") // class hkMemoryAllocator*
    HK_TRACKER_MEMBER(hkTempDetectAllocator, m_callTree, 0, "hkStackTracer::CallTree") // class hkStackTracer::CallTree
    HK_TRACKER_MEMBER(hkTempDetectAllocator, m_tracer, 0, "hkStackTracer") // class hkStackTracer
    HK_TRACKER_MEMBER(hkTempDetectAllocator, m_allocs, 0, "hkMapBase<void*, hkTempDetectAllocator::AllocInfo, hkMapOperations<void*> >") // class hkMapBase< void*, struct hkTempDetectAllocator::AllocInfo, struct hkMapOperations< void* > >
    HK_TRACKER_MEMBER(hkTempDetectAllocator, m_freeFromAlloc, 0, "hkMapBase<hkInt32, hkInt32, hkMapOperations<hkInt32> >") // class hkMapBase< hkInt32, hkInt32, struct hkMapOperations< hkInt32 > >
    HK_TRACKER_MEMBER(hkTempDetectAllocator, m_sizeFromAlloc, 0, "hkMapBase<hkInt32, hkTempDetectAllocator::SizeInfo, hkMapOperations<hkInt32> >") // class hkMapBase< hkInt32, struct hkTempDetectAllocator::SizeInfo, struct hkMapOperations< hkInt32 > >
    HK_TRACKER_MEMBER(hkTempDetectAllocator, m_outputFuncArg, 0, "void*") // void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkTempDetectAllocator, s_libraryName, hkMemoryAllocator)


// AllocInfo hkTempDetectAllocator
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkTempDetectAllocator, AllocInfo, s_libraryName)


// SizeInfo hkTempDetectAllocator
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkTempDetectAllocator, SizeInfo, s_libraryName)

#include <Common/Base/Memory/Allocator/Thread/hkThreadMemory.h>

// hk.MemoryTracker ignore hkThreadMemory
#include <Common/Base/Memory/Allocator/hkMemoryAllocator.h>

 // Skipping Class hkSizeOfTypeOrVoid< void > as it is a template

// hk.MemoryTracker ignore hkMemoryAllocator
#include <Common/Base/Memory/MemoryClasses/hkMemoryClassesTable.h>


// hkMemoryClassStatistics ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkMemoryClassStatistics, s_libraryName)


// hkMemoryClassInfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMemoryClassInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMemoryClassInfo)
    HK_TRACKER_MEMBER(hkMemoryClassInfo, m_name, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkMemoryClassInfo, s_libraryName)

#include <Common/Base/Memory/Router/hkMemoryRouter.h>

// hk.MemoryTracker ignore hkMemoryRouter
#include <Common/Base/Memory/System/Checking/hkCheckingMemorySystem.h>

// hk.MemoryTracker ignore hkCheckingMemorySystem
#include <Common/Base/Memory/System/Debug/hkDebugMemorySystem.h>


// hkDebugMemorySystem ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDebugMemorySystem)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BaseAddressResuls)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDebugMemorySystem)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkDebugMemorySystem, s_libraryName, hkMemorySystem)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkDebugMemorySystem, BaseAddressResuls, s_libraryName)

#include <Common/Base/Memory/System/FreeList/hkFreeListMemorySystem.h>

// hk.MemoryTracker ignore hkFreeListMemorySystem
#include <Common/Base/Memory/System/Optimizer/hkOptimizerMemorySystem.h>

// hk.MemoryTracker ignore hkOptimizerMemorySystem
#include <Common/Base/Memory/System/Simple/hkSimpleMemorySystem.h>

// hk.MemoryTracker ignore hkSimpleMemorySystem
#include <Common/Base/Memory/System/Util/hkMemoryInitUtil.h>


// SyncInfo hkMemoryInitUtil

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMemoryInitUtil::SyncInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMemoryInitUtil::SyncInfo)
    HK_TRACKER_MEMBER(hkMemoryInitUtil::SyncInfo, m_memoryRouter, 0, "hkMemoryRouter*") // class hkMemoryRouter*
    HK_TRACKER_MEMBER(hkMemoryInitUtil::SyncInfo, m_memorySystem, 0, "hkMemorySystem*") // class hkMemorySystem*
    HK_TRACKER_MEMBER(hkMemoryInitUtil::SyncInfo, m_singletonList, 0, "hkSingletonInitNode*") // struct hkSingletonInitNode*
    HK_TRACKER_MEMBER(hkMemoryInitUtil::SyncInfo, m_stackTracerImpl, 0, "void*") // void*
    HK_TRACKER_MEMBER(hkMemoryInitUtil::SyncInfo, m_mtCheckSection, 0, "hkCriticalSection*") // class hkCriticalSection*
    HK_TRACKER_MEMBER(hkMemoryInitUtil::SyncInfo, m_monitors, 0, "hkMonitorStream*") // class hkMonitorStream*
    HK_TRACKER_MEMBER(hkMemoryInitUtil::SyncInfo, m_mtCheckStackTree, 0, "hkStackTracer::CallTree*") // class hkStackTracer::CallTree*
    HK_TRACKER_MEMBER(hkMemoryInitUtil::SyncInfo, m_mtRefLockedAllPtr, 0, "hkUint32*") // hkUint32*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkMemoryInitUtil::SyncInfo, s_libraryName)

#include <Common/Base/Memory/System/Util/hkMemorySnapshot.h>


// hkMemorySnapshot ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMemorySnapshot)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Allocation)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Provider)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(StatusBits)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMemorySnapshot)
    HK_TRACKER_MEMBER(hkMemorySnapshot, m_mem, 0, "hkMemoryAllocator*") // class hkMemoryAllocator*
    HK_TRACKER_MEMBER(hkMemorySnapshot, m_allocations, 0, "hkArrayBase<hkMemorySnapshot::Allocation>") // class hkArrayBase< struct hkMemorySnapshot::Allocation >
    HK_TRACKER_MEMBER(hkMemorySnapshot, m_providers, 0, "hkArrayBase<hkMemorySnapshot::Provider>") // class hkArrayBase< struct hkMemorySnapshot::Provider >
    HK_TRACKER_MEMBER(hkMemorySnapshot, m_callTree, 0, "hkStackTracer::CallTree") // class hkStackTracer::CallTree
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkMemorySnapshot, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkMemorySnapshot, StatusBits, s_libraryName)


// Allocation hkMemorySnapshot

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMemorySnapshot::Allocation)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMemorySnapshot::Allocation)
    HK_TRACKER_MEMBER(hkMemorySnapshot::Allocation, m_start, 0, "void*") // const void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkMemorySnapshot::Allocation, s_libraryName)


// Provider hkMemorySnapshot

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMemorySnapshot::Provider)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMemorySnapshot::Provider)
    HK_TRACKER_MEMBER(hkMemorySnapshot::Provider, m_parentIndices, 0, "hkArrayBase<hkInt32>") // class hkArrayBase< hkInt32 >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkMemorySnapshot::Provider, s_libraryName)

#include <Common/Base/Memory/System/hkMemorySystem.h>

// hk.MemoryTracker ignore hkMemorySystem
#include <Common/Base/Memory/Tracker/CurrentFunction/hkCurrentFunction.h>


// hkCurrentFunctionUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkCurrentFunctionUtil, s_libraryName)

#include <Common/Base/Memory/Tracker/Default/hkDefaultMemoryTracker.h>


// hkDefaultMemoryTrackerAllocator ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkDefaultMemoryTrackerAllocator, s_libraryName)


// hkDefaultMemoryTracker ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDefaultMemoryTracker)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ClassAlloc)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDefaultMemoryTracker)
    HK_TRACKER_MEMBER(hkDefaultMemoryTracker, m_deletedMap, 0, "hkPointerMap<void*, hkInt32, hkDefaultMemoryTrackerAllocator>") // class hkPointerMap< void*, hkInt32, struct hkDefaultMemoryTrackerAllocator >
    HK_TRACKER_MEMBER(hkDefaultMemoryTracker, m_createdMap, 0, "hkPointerMap<void*, hkDefaultMemoryTracker::ClassAlloc*, hkDefaultMemoryTrackerAllocator>") // class hkPointerMap< void*, struct hkDefaultMemoryTracker::ClassAlloc*, struct hkDefaultMemoryTrackerAllocator >
    HK_TRACKER_MEMBER(hkDefaultMemoryTracker, m_classAllocFreeList, 0, "hkFreeList") // class hkFreeList
    HK_TRACKER_MEMBER(hkDefaultMemoryTracker, m_classAllocMap, 0, "hkPointerMap<void*, hkDefaultMemoryTracker::ClassAlloc*, hkDefaultMemoryTrackerAllocator>") // class hkPointerMap< void*, struct hkDefaultMemoryTracker::ClassAlloc*, struct hkDefaultMemoryTrackerAllocator >
    HK_TRACKER_MEMBER(hkDefaultMemoryTracker, m_nameTypeMap, 0, "hkStringMap<hkMemoryTracker::TypeDefinition*, hkDefaultMemoryTrackerAllocator>") // class hkStringMap< const struct hkMemoryTracker::TypeDefinition*, struct hkDefaultMemoryTrackerAllocator >
    HK_TRACKER_MEMBER(hkDefaultMemoryTracker, m_assertOnRemove, 0, "hkDefaultMemoryTracker::ClassAlloc*") // const struct hkDefaultMemoryTracker::ClassAlloc*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkDefaultMemoryTracker, s_libraryName, hkMemoryTracker)


// ClassAlloc hkDefaultMemoryTracker

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDefaultMemoryTracker::ClassAlloc)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDefaultMemoryTracker::ClassAlloc)
    HK_TRACKER_MEMBER(hkDefaultMemoryTracker::ClassAlloc, m_typeName, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkDefaultMemoryTracker::ClassAlloc, m_ptr, 0, "void*") // void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkDefaultMemoryTracker::ClassAlloc, s_libraryName)

#include <Common/Base/Memory/Tracker/LayoutCalculator/Handler/hkTrackerArrayLayoutHandler.h>

// hk.MemoryTracker ignore hkTrackerArrayLayoutHandler
#include <Common/Base/Memory/Tracker/LayoutCalculator/Handler/hkTrackerEnumLayoutHandler.h>

// hk.MemoryTracker ignore hkTrackerEnumLayoutHandler
#include <Common/Base/Memory/Tracker/LayoutCalculator/Handler/hkTrackerFlagsLayoutHandler.h>

// hk.MemoryTracker ignore hkTrackerFlagsLayoutHandler
#include <Common/Base/Memory/Tracker/LayoutCalculator/Handler/hkTrackerJobQueueLayoutHandler.h>

// hk.MemoryTracker ignore hkTrackerJobQueueLayoutHandler
// hk.MemoryTracker ignore hkTrackerJobQueueDynamicDataLayoutHandler
#include <Common/Base/Memory/Tracker/LayoutCalculator/Handler/hkTrackerPadSpuLayoutHandler.h>

// hk.MemoryTracker ignore hkTrackerPadSpuLayoutHandler
#include <Common/Base/Memory/Tracker/LayoutCalculator/Handler/hkTrackerPointerMapLayoutHandler.h>

// hk.MemoryTracker ignore hkTrackerPointerMapLayoutHandler
#include <Common/Base/Memory/Tracker/LayoutCalculator/Handler/hkTrackerQueueLayoutHandler.h>

// hk.MemoryTracker ignore hkTrackerQueueLayoutHandler
#include <Common/Base/Memory/Tracker/LayoutCalculator/Handler/hkTrackerRefPtrLayoutHandler.h>

// hk.MemoryTracker ignore hkTrackerRefPtrLayoutHandler
#include <Common/Base/Memory/Tracker/LayoutCalculator/Handler/hkTrackerSetLayoutHandler.h>

// hk.MemoryTracker ignore hkTrackerSetLayoutHandler
#include <Common/Base/Memory/Tracker/LayoutCalculator/Handler/hkTrackerStringMapLayoutHandler.h>

// hk.MemoryTracker ignore hkTrackerStringMapLayoutHandler
#include <Common/Base/Memory/Tracker/LayoutCalculator/Handler/hkTrackerStringPtrLayoutHandler.h>

// hk.MemoryTracker ignore hkTrackerStringPtrLayoutHandler
#include <Common/Base/Memory/Tracker/LayoutCalculator/hkTrackerExternalLayoutHandlerManager.h>


// hkTrackerExternalLayoutHandlerManager ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkTrackerExternalLayoutHandlerManager)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkTrackerExternalLayoutHandlerManager)
    HK_TRACKER_MEMBER(hkTrackerExternalLayoutHandlerManager, m_handlers, 0, "hkStorageStringMap<hkTrackerLayoutHandler*, hkContainerHeapAllocator>") // class hkStorageStringMap< class hkTrackerLayoutHandler*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkTrackerExternalLayoutHandlerManager, s_libraryName, hkReferencedObject)

#include <Common/Base/Memory/Tracker/LayoutCalculator/hkTrackerLayoutCalculator.h>


// hkTrackerTypeLayout ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkTrackerTypeLayout)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Member)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkTrackerTypeLayout)
    HK_TRACKER_MEMBER(hkTrackerTypeLayout, m_type, 0, "hkTrackerTypeTreeNode*") // const class hkTrackerTypeTreeNode*
    HK_TRACKER_MEMBER(hkTrackerTypeLayout, m_members, 0, "hkArray<hkTrackerTypeLayout::Member, hkContainerHeapAllocator>") // hkArray< struct hkTrackerTypeLayout::Member, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkTrackerTypeLayout, m_nameBuffer, 0, "hkArray<char, hkContainerHeapAllocator>") // hkArray< char, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkTrackerTypeLayout, s_libraryName, hkReferencedObject)


// Member hkTrackerTypeLayout

HK_TRACKER_DECLARE_CLASS_BEGIN(hkTrackerTypeLayout::Member)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkTrackerTypeLayout::Member)
    HK_TRACKER_MEMBER(hkTrackerTypeLayout::Member, m_type, 0, "hkTrackerTypeTreeNode*") // const class hkTrackerTypeTreeNode*
    HK_TRACKER_MEMBER(hkTrackerTypeLayout::Member, m_name, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkTrackerTypeLayout::Member, s_libraryName)


// hkTrackerLayoutTypeInfo ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkTrackerLayoutTypeInfo, s_libraryName)


// hkTrackerLayoutBlock ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkTrackerLayoutBlock)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkTrackerLayoutBlock)
    HK_TRACKER_MEMBER(hkTrackerLayoutBlock, m_type, 0, "hkTrackerTypeTreeNode*") // const class hkTrackerTypeTreeNode*
    HK_TRACKER_MEMBER(hkTrackerLayoutBlock, m_start, 0, "void*") // const void*
    HK_TRACKER_MEMBER(hkTrackerLayoutBlock, m_references, 0, "hkArray<void*, hkContainerHeapAllocator>") // hkArray< const void*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkTrackerLayoutBlock, s_libraryName)

// hk.MemoryTracker ignore hkTrackerLayoutHandler
// hk.MemoryTracker ignore hkTrackerLayoutCalculator
#include <Common/Base/Memory/Tracker/Report/hkCategoryReportUtil.h>


// hkCategoryReportUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkCategoryReportUtil, s_libraryName)

#include <Common/Base/Memory/Tracker/Report/hkFieldHierarchyReportUtil.h>


// hkFieldHierarchyReportUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkFieldHierarchyReportUtil, s_libraryName)

#include <Common/Base/Memory/Tracker/Report/hkHierarchicalSummaryReportUtil.h>


// hkHierarchicalSummaryReportUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkHierarchicalSummaryReportUtil, s_libraryName)

#include <Common/Base/Memory/Tracker/Report/hkHierarchyReportUtil.h>


// hkHierarchyReportUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkHierarchyReportUtil, s_libraryName)

#include <Common/Base/Memory/Tracker/Report/hkProductReportUtil.h>


// hkProductReportUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkProductReportUtil, s_libraryName)

#include <Common/Base/Memory/Tracker/Report/hkScanReportUtil.h>


// hkScanReportUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkScanReportUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MemorySize)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(NameTypePair)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(FollowFilter)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Traversal)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkScanReportUtil, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkScanReportUtil, Traversal, s_libraryName)


// MemorySize hkScanReportUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkScanReportUtil, MemorySize, s_libraryName)


// NameTypePair hkScanReportUtil

HK_TRACKER_DECLARE_CLASS_BEGIN(hkScanReportUtil::NameTypePair)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkScanReportUtil::NameTypePair)
    HK_TRACKER_MEMBER(hkScanReportUtil::NameTypePair, m_name, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkScanReportUtil::NameTypePair, s_libraryName)


// FollowFilter hkScanReportUtil

HK_TRACKER_DECLARE_CLASS_BEGIN(hkScanReportUtil::FollowFilter)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkScanReportUtil::FollowFilter)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkScanReportUtil::FollowFilter, s_libraryName)

#include <Common/Base/Memory/Tracker/Report/hkStackTraceReportUtil.h>


// hkStackTraceReportUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkStackTraceReportUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Summary)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkStackTraceReportUtil, s_libraryName)


// Summary hkStackTraceReportUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkStackTraceReportUtil, Summary, s_libraryName)

#include <Common/Base/Memory/Tracker/Report/hkTulipReportUtil.h>


// hkTulipReportUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkTulipReportUtil, s_libraryName)

#include <Common/Base/Memory/Tracker/Report/hkTypeReportUtil.h>


// hkTypeReportUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkTypeReportUtil, s_libraryName)

#include <Common/Base/Memory/Tracker/Report/hkTypeSummaryReportUtil.h>


// hkTypeSummaryReportUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkTypeSummaryReportUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(TypeSummary)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkTypeSummaryReportUtil, s_libraryName)


// TypeSummary hkTypeSummaryReportUtil

HK_TRACKER_DECLARE_CLASS_BEGIN(hkTypeSummaryReportUtil::TypeSummary)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkTypeSummaryReportUtil::TypeSummary)
    HK_TRACKER_MEMBER(hkTypeSummaryReportUtil::TypeSummary, m_type, 0, "hkTrackerTypeTreeNode*") // const class hkTrackerTypeTreeNode*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkTypeSummaryReportUtil::TypeSummary, s_libraryName)

#include <Common/Base/Memory/Tracker/Report/hkVdbStreamReportUtil.h>


// hkVdbStreamReportUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkVdbStreamReportUtil, s_libraryName)

#include <Common/Base/Memory/Tracker/ScanCalculator/hkTrackerScanCalculator.h>


// hkTrackerScanCalculator ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkTrackerScanCalculator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkTrackerScanCalculator)
    HK_TRACKER_MEMBER(hkTrackerScanCalculator, m_scanSnapshot, 0, "hkTrackerScanSnapshot*") // class hkTrackerScanSnapshot*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkTrackerScanCalculator, s_libraryName)

#include <Common/Base/Memory/Tracker/ScanCalculator/hkTrackerScanSnapshot.h>

// hk.MemoryTracker ignore hkTrackerScanSnapshot
#include <Common/Base/Memory/Tracker/ScanCalculator/hkTrackerSnapshot.h>

// hk.MemoryTracker ignore hkTrackerSnapshot
#include <Common/Base/Memory/Tracker/ScanCalculator/hkTrackerSnapshotUtil.h>


// hkTrackerSnapshotUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkTrackerSnapshotUtil, s_libraryName)

#include <Common/Base/Memory/Tracker/Util/hkTextReportLayoutUtil.h>

// hk.MemoryTracker ignore hkTextReportLayoutUtil
#include <Common/Base/Memory/Tracker/Util/hkVerifySnapshotTypesUtil.h>


// hkVerifySnapshotTypesUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkVerifySnapshotTypesUtil, s_libraryName)

#include <Common/Base/Memory/Tracker/hkMemoryTracker.h>

// hk.MemoryTracker ignore hkMemoryTracker
#include <Common/Base/Memory/Tracker/hkTrackerClassDefinition.h>


// hkTrackerTypeInit ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkTrackerTypeInit)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkTrackerTypeInit)
    HK_TRACKER_MEMBER(hkTrackerTypeInit, m_next, 0, "hkTrackerTypeInit*") // class hkTrackerTypeInit*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkTrackerTypeInit, s_libraryName)

#include <Common/Base/Memory/Util/hkMemUtil.h>

 // Skipping Class TypeFromAlign< 1 > as it is a template

 // Skipping Class TypeFromAlign< 2 > as it is a template

 // Skipping Class TypeFromAlign< 4 > as it is a template

 // Skipping Class TypeFromAlign< 8 > as it is a template

#include <Common/Base/Monitor/hkMonitorStream.h>


// hkTimerData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkTimerData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkTimerData)
    HK_TRACKER_MEMBER(hkTimerData, m_streamBegin, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkTimerData, m_streamEnd, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkTimerData, s_libraryName)


// hkMonitorStream ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMonitorStream)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Command)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(AddValueCommand)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(TimerCommand)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(TimerBeginListCommand)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(TimerBeginObjectNameCommand)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MemoryCommand)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMonitorStream)
    HK_TRACKER_MEMBER(hkMonitorStream, m_start, 0, "hkPadSpu<char*>") // class hkPadSpu< char* >
    HK_TRACKER_MEMBER(hkMonitorStream, m_end, 0, "hkPadSpu<char*>") // class hkPadSpu< char* >
    HK_TRACKER_MEMBER(hkMonitorStream, m_capacity, 0, "hkPadSpu<char*>") // class hkPadSpu< char* >
    HK_TRACKER_MEMBER(hkMonitorStream, m_capacityMinus16, 0, "hkPadSpu<char*>") // class hkPadSpu< char* >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkMonitorStream, s_libraryName)


// Command hkMonitorStream

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMonitorStream::Command)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMonitorStream::Command)
    HK_TRACKER_MEMBER(hkMonitorStream::Command, m_commandAndMonitor, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkMonitorStream::Command, s_libraryName)


// AddValueCommand hkMonitorStream
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkMonitorStream, AddValueCommand, s_libraryName)


// TimerCommand hkMonitorStream
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkMonitorStream, TimerCommand, s_libraryName)


// TimerBeginListCommand hkMonitorStream

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMonitorStream::TimerBeginListCommand)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMonitorStream::TimerBeginListCommand)
    HK_TRACKER_MEMBER(hkMonitorStream::TimerBeginListCommand, m_nameOfFirstSplit, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkMonitorStream::TimerBeginListCommand, s_libraryName, hkMonitorStream::TimerCommand)


// TimerBeginObjectNameCommand hkMonitorStream

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMonitorStream::TimerBeginObjectNameCommand)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMonitorStream::TimerBeginObjectNameCommand)
    HK_TRACKER_MEMBER(hkMonitorStream::TimerBeginObjectNameCommand, m_objectName, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkMonitorStream::TimerBeginObjectNameCommand, s_libraryName, hkMonitorStream::TimerCommand)


// MemoryCommand hkMonitorStream

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMonitorStream::MemoryCommand)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMonitorStream::MemoryCommand)
    HK_TRACKER_MEMBER(hkMonitorStream::MemoryCommand, m_ptr, 0, "void*") // void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkMonitorStream::MemoryCommand, s_libraryName, hkMonitorStream::Command)

#include <Common/Base/Object/hkSingleton.h>


// hkSingletonInitNode ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkSingletonInitNode)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkSingletonInitNode)
    HK_TRACKER_MEMBER(hkSingletonInitNode, m_name, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkSingletonInitNode, m_next, 0, "hkSingletonInitNode*") // struct hkSingletonInitNode*
    HK_TRACKER_MEMBER(hkSingletonInitNode, m_value, 0, "void**") // void**
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkSingletonInitNode, s_libraryName)

#include <Common/Base/Reflection/Attributes/hkAttributes.h>


// hkRangeRealAttribute ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkRangeRealAttribute, s_libraryName)


// hkRangeInt32Attribute ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkRangeInt32Attribute, s_libraryName)


// hkUiAttribute ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkUiAttribute)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(HideInModeler)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkUiAttribute)
    HK_TRACKER_MEMBER(hkUiAttribute, m_label, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkUiAttribute, m_group, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkUiAttribute, m_hideBaseClassMembers, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkUiAttribute, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkUiAttribute, HideInModeler, s_libraryName)


// hkGizmoAttribute ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkGizmoAttribute)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(GizmoType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkGizmoAttribute)
    HK_TRACKER_MEMBER(hkGizmoAttribute, m_label, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkGizmoAttribute, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkGizmoAttribute, GizmoType, s_libraryName)


// hkModelerNodeTypeAttribute ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkModelerNodeTypeAttribute, s_libraryName)


// hkLinkAttribute ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkLinkAttribute, s_libraryName)


// hkSemanticsAttribute ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkSemanticsAttribute, s_libraryName)


// hkDescriptionAttribute ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDescriptionAttribute)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDescriptionAttribute)
    HK_TRACKER_MEMBER(hkDescriptionAttribute, m_string, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkDescriptionAttribute, s_libraryName)


// hkArrayTypeAttribute ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkArrayTypeAttribute, s_libraryName)


// hkDataObjectTypeAttribute ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDataObjectTypeAttribute)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDataObjectTypeAttribute)
    HK_TRACKER_MEMBER(hkDataObjectTypeAttribute, m_typeName, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkDataObjectTypeAttribute, s_libraryName)


// hkDocumentationAttribute ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDocumentationAttribute)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDocumentationAttribute)
    HK_TRACKER_MEMBER(hkDocumentationAttribute, m_docsSectionTag, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkDocumentationAttribute, s_libraryName)


// hkPostFinishAttribute ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkPostFinishAttribute, s_libraryName)


// hkScriptableAttribute ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkScriptableAttribute, s_libraryName)

#include <Common/Base/Reflection/Registry/hkClassNameRegistry.h>


// hkClassNameRegistry ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkClassNameRegistry)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkClassNameRegistry)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkClassNameRegistry, s_libraryName, hkReferencedObject)

#include <Common/Base/Reflection/Registry/hkDefaultClassNameRegistry.h>


// hkDefaultClassNameRegistry ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDefaultClassNameRegistry)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDefaultClassNameRegistry)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkDefaultClassNameRegistry, s_libraryName, hkDynamicClassNameRegistry)

#include <Common/Base/Reflection/Registry/hkDynamicClassNameRegistry.h>


// hkDynamicClassNameRegistry ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDynamicClassNameRegistry)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDynamicClassNameRegistry)
    HK_TRACKER_MEMBER(hkDynamicClassNameRegistry, m_name, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkDynamicClassNameRegistry, m_map, 0, "hkStringMap<hkClass*, hkContainerHeapAllocator>") // class hkStringMap< const hkClass*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkDynamicClassNameRegistry, s_libraryName, hkClassNameRegistry)

#include <Common/Base/Reflection/Registry/hkTypeInfoRegistry.h>


// hkTypeInfoRegistry ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkTypeInfoRegistry)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkTypeInfoRegistry)
    HK_TRACKER_MEMBER(hkTypeInfoRegistry, m_map, 0, "hkStringMap<hkTypeInfo*, hkContainerHeapAllocator>") // class hkStringMap< const class hkTypeInfo*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkTypeInfoRegistry, s_libraryName, hkReferencedObject)

#include <Common/Base/Reflection/Registry/hkVtableClassRegistry.h>


// hkVtableClassRegistry ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkVtableClassRegistry)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkVtableClassRegistry)
    HK_TRACKER_MEMBER(hkVtableClassRegistry, m_map, 0, "hkPointerMap<void*, hkClass*, hkContainerHeapAllocator>") // class hkPointerMap< const void*, const hkClass*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkVtableClassRegistry, s_libraryName, hkReferencedObject)

#include <Common/Base/Reflection/TypeTree/hkTrackerTypeTreeCache.h>


// hkTrackerTypeTreeCache ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkTrackerTypeTreeCache)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkTrackerTypeTreeCache)
    HK_TRACKER_MEMBER(hkTrackerTypeTreeCache, m_builtInTypes, 0, "hkTrackerTypeTreeNode* [21]") // const class hkTrackerTypeTreeNode* [21]
    HK_TRACKER_MEMBER(hkTrackerTypeTreeCache, m_nodeFreeList, 0, "hkFreeList") // class hkFreeList
    HK_TRACKER_MEMBER(hkTrackerTypeTreeCache, m_textMap, 0, "hkStorageStringMap<hkInt32, hkContainerHeapAllocator>") // class hkStorageStringMap< hkInt32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkTrackerTypeTreeCache, m_namedTypeMap, 0, "hkStringMap<hkTrackerTypeTreeNode*, hkContainerHeapAllocator>") // class hkStringMap< const class hkTrackerTypeTreeNode*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkTrackerTypeTreeCache, m_expressionTypeMap, 0, "hkStringMap<hkTrackerTypeTreeNode*, hkContainerHeapAllocator>") // class hkStringMap< const class hkTrackerTypeTreeNode*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkTrackerTypeTreeCache, s_libraryName, hkReferencedObject)

#include <Common/Base/Reflection/TypeTree/hkTrackerTypeTreeNode.h>


// hkTrackerTypeTreeNode ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkTrackerTypeTreeNode)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Type)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkTrackerTypeTreeNode)
    HK_TRACKER_MEMBER(hkTrackerTypeTreeNode, m_name, 0, "hkSubString") // struct hkSubString
    HK_TRACKER_MEMBER(hkTrackerTypeTreeNode, m_contains, 0, "hkTrackerTypeTreeNode*") // const class hkTrackerTypeTreeNode*
    HK_TRACKER_MEMBER(hkTrackerTypeTreeNode, m_next, 0, "hkTrackerTypeTreeNode*") // const class hkTrackerTypeTreeNode*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkTrackerTypeTreeNode, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkTrackerTypeTreeNode, Type, s_libraryName)

#include <Common/Base/Reflection/TypeTree/hkTrackerTypeTreeParser.h>

// hk.MemoryTracker ignore hkTrackerTypeTreeParser
#include <Common/Base/Reflection/TypeTree/hkTrackerTypeTreeTextCache.h>


// hkTrackerTypeTreeTextCache ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkTrackerTypeTreeTextCache)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkTrackerTypeTreeTextCache)
    HK_TRACKER_MEMBER(hkTrackerTypeTreeTextCache, m_typeNames, 0, "hkArray<char, hkContainerHeapAllocator>") // hkArray< char, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkTrackerTypeTreeTextCache, m_map, 0, "hkPointerMap<hkTrackerTypeTreeNode*, hkInt32, hkContainerHeapAllocator>") // class hkPointerMap< const class hkTrackerTypeTreeNode*, hkInt32, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkTrackerTypeTreeTextCache, s_libraryName, hkReferencedObject)

#include <Common/Base/Reflection/Util/hkVariantDataUtil.h>


// hkStridedBasicArray ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkStridedBasicArray)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkStridedBasicArray)
    HK_TRACKER_MEMBER(hkStridedBasicArray, m_data, 0, "void*") // void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkStridedBasicArray, s_libraryName)


// hkVariantDataUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkVariantDataUtil, s_libraryName)

#include <Common/Base/Reflection/hkClass.h>


// hkClass ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkClass)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SignatureFlags)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(FlagValues)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkClass)
    HK_TRACKER_MEMBER(hkClass, m_name, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkClass, m_parent, 0, "hkClass*") // const hkClass*
    HK_TRACKER_MEMBER(hkClass, m_defaults, 0, "void*") // const void*
    HK_TRACKER_MEMBER(hkClass, m_attributes, 0, "hkCustomAttributes*") // const class hkCustomAttributes*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkClass, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkClass, SignatureFlags, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkClass, FlagValues, s_libraryName)

#include <Common/Base/Reflection/hkClassEnum.h>


// hkClassEnum ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkClassEnum)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Item)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(FlagValues)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkClassEnum)
    HK_TRACKER_MEMBER(hkClassEnum, m_name, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkClassEnum, m_attributes, 0, "hkCustomAttributes*") // class hkCustomAttributes*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkClassEnum, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkClassEnum, FlagValues, s_libraryName)


// Item hkClassEnum

HK_TRACKER_DECLARE_CLASS_BEGIN(hkClassEnum::Item)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkClassEnum::Item)
    HK_TRACKER_MEMBER(hkClassEnum::Item, m_name, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkClassEnum::Item, s_libraryName)

#include <Common/Base/Reflection/hkClassMember.h>


// hkClassMember ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkClassMember)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(TypeProperties)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Type)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(FlagValues)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(DeprecatedFlagValues)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkClassMember)
    HK_TRACKER_MEMBER(hkClassMember, m_name, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkClassMember, m_class, 0, "hkClass*") // const hkClass*
    HK_TRACKER_MEMBER(hkClassMember, m_enum, 0, "hkClassEnum*") // const class hkClassEnum*
    HK_TRACKER_MEMBER(hkClassMember, m_attributes, 0, "hkCustomAttributes*") // const class hkCustomAttributes*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkClassMember, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkClassMember, Type, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkClassMember, FlagValues, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkClassMember, DeprecatedFlagValues, s_libraryName)


// TypeProperties hkClassMember

HK_TRACKER_DECLARE_CLASS_BEGIN(hkClassMember::TypeProperties)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkClassMember::TypeProperties)
    HK_TRACKER_MEMBER(hkClassMember::TypeProperties, m_name, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkClassMember::TypeProperties, s_libraryName)

#include <Common/Base/Reflection/hkClassMemberAccessor.h>


// hkClassMemberAccessor ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkClassMemberAccessor)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SimpleArray)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(HomogeneousArray)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Vector4)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Matrix3)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Transform)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkClassMemberAccessor)
    HK_TRACKER_MEMBER(hkClassMemberAccessor, m_address, 0, "void*") // void*
    HK_TRACKER_MEMBER(hkClassMemberAccessor, m_member, 0, "hkClassMember*") // const class hkClassMember*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkClassMemberAccessor, s_libraryName)


// SimpleArray hkClassMemberAccessor

HK_TRACKER_DECLARE_CLASS_BEGIN(hkClassMemberAccessor::SimpleArray)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkClassMemberAccessor::SimpleArray)
    HK_TRACKER_MEMBER(hkClassMemberAccessor::SimpleArray, data, 0, "void*") // void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkClassMemberAccessor::SimpleArray, s_libraryName)


// HomogeneousArray hkClassMemberAccessor

HK_TRACKER_DECLARE_CLASS_BEGIN(hkClassMemberAccessor::HomogeneousArray)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkClassMemberAccessor::HomogeneousArray)
    HK_TRACKER_MEMBER(hkClassMemberAccessor::HomogeneousArray, klass, 0, "hkClass*") // hkClass*
    HK_TRACKER_MEMBER(hkClassMemberAccessor::HomogeneousArray, data, 0, "void*") // void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkClassMemberAccessor::HomogeneousArray, s_libraryName)


// Vector4 hkClassMemberAccessor
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkClassMemberAccessor, Vector4, s_libraryName)


// Matrix3 hkClassMemberAccessor
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkClassMemberAccessor, Matrix3, s_libraryName)


// Transform hkClassMemberAccessor
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkClassMemberAccessor, Transform, s_libraryName)


// hkClassAccessor ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkClassAccessor)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkClassAccessor)
    HK_TRACKER_MEMBER(hkClassAccessor, m_variant, 0, "hkVariant") // hkVariant
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkClassAccessor, s_libraryName)

#include <Common/Base/Reflection/hkCustomAttributes.h>


// hkCustomAttributes ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkCustomAttributes)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Attribute)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkCustomAttributes, s_libraryName)


// Attribute hkCustomAttributes

HK_TRACKER_DECLARE_CLASS_BEGIN(hkCustomAttributes::Attribute)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkCustomAttributes::Attribute)
    HK_TRACKER_MEMBER(hkCustomAttributes::Attribute, m_name, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkCustomAttributes::Attribute, m_value, 0, "hkVariant") // hkVariant
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkCustomAttributes::Attribute, s_libraryName)

#include <Common/Base/Reflection/hkInternalClassMember.h>


// hkInternalCustomAttributes ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkInternalCustomAttributes)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Attribute)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkInternalCustomAttributes, s_libraryName)


// Attribute hkInternalCustomAttributes

HK_TRACKER_DECLARE_CLASS_BEGIN(hkInternalCustomAttributes::Attribute)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkInternalCustomAttributes::Attribute)
    HK_TRACKER_MEMBER(hkInternalCustomAttributes::Attribute, m_name, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkInternalCustomAttributes::Attribute, m_data, 0, "void*") // const void*
    HK_TRACKER_MEMBER(hkInternalCustomAttributes::Attribute, m_klass, 0, "hkClass*") // const hkClass*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkInternalCustomAttributes::Attribute, s_libraryName)


// hkInternalClassMember ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkInternalClassMember)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkInternalClassMember)
    HK_TRACKER_MEMBER(hkInternalClassMember, m_name, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkInternalClassMember, m_class, 0, "hkClass*") // const hkClass*
    HK_TRACKER_MEMBER(hkInternalClassMember, m_enum, 0, "hkClassEnum*") // const class hkClassEnum*
    HK_TRACKER_MEMBER(hkInternalClassMember, m_attributes, 0, "hkInternalCustomAttributes*") // const struct hkInternalCustomAttributes*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkInternalClassMember, s_libraryName)


// hkInternalClassEnumItem ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkInternalClassEnumItem)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkInternalClassEnumItem)
    HK_TRACKER_MEMBER(hkInternalClassEnumItem, m_name, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkInternalClassEnumItem, s_libraryName)


// hkInternalClassEnum ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkInternalClassEnum)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkInternalClassEnum)
    HK_TRACKER_MEMBER(hkInternalClassEnum, m_name, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkInternalClassEnum, m_attributes, 0, "hkInternalCustomAttributes*") // const struct hkInternalCustomAttributes*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkInternalClassEnum, s_libraryName)


// hkInternalClass ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkInternalClass)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkInternalClass)
    HK_TRACKER_MEMBER(hkInternalClass, m_name, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkInternalClass, m_parent, 0, "hkInternalClass*") // const struct hkInternalClass*
    HK_TRACKER_MEMBER(hkInternalClass, m_defaults, 0, "void*") // const void*
    HK_TRACKER_MEMBER(hkInternalClass, m_attributes, 0, "hkInternalCustomAttributes*") // const struct hkInternalCustomAttributes*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkInternalClass, s_libraryName)

#include <Common/Base/Reflection/hkTypeInfo.h>


// hkTypeInfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkTypeInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkTypeInfo)
    HK_TRACKER_MEMBER(hkTypeInfo, m_typeName, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkTypeInfo, m_scopedName, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkTypeInfo, m_vtable, 0, "void*") // const void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkTypeInfo, s_libraryName)

#include <Common/Base/System/Error/hkDefaultError.h>


// hkDefaultError ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDefaultError)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDefaultError)
    HK_TRACKER_MEMBER(hkDefaultError, m_disabledAssertIds, 0, "hkPointerMap<hkInt32, hkInt32, hkContainerHeapAllocator>") // class hkPointerMap< hkInt32, hkInt32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkDefaultError, m_sectionIds, 0, "hkArray<hkInt32, hkContainerHeapAllocator>") // hkArray< hkInt32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkDefaultError, m_errorObject, 0, "void*") // void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkDefaultError, s_libraryName, hkError)

#include <Common/Base/System/Error/hkError.h>


// hkError ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkError)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Message)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkError)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkError, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkError, Message, s_libraryName)


// hkDisableError ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkDisableError, s_libraryName)


// hkErrStream ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkErrStream)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkErrStream)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkErrStream, s_libraryName, hkOstream)

 // Skipping Class COMPILE_ASSERTION_FAILURE< 1 > as it is a template

 // Skipping Class REFLECTION_PARSER_VTABLE_DETECTION_FAILED< 1 > as it is a template

#include <Common/Base/System/Io/FileSystem/Union/hkUnionFileSystem.h>


// hkUnionFileSystem ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkUnionFileSystem)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Mount)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkUnionFileSystem)
    HK_TRACKER_MEMBER(hkUnionFileSystem, m_mounts, 0, "hkArray<hkUnionFileSystem::Mount, hkContainerHeapAllocator>") // hkArray< struct hkUnionFileSystem::Mount, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkUnionFileSystem, s_libraryName, hkFileSystem)


// Mount hkUnionFileSystem

HK_TRACKER_DECLARE_CLASS_BEGIN(hkUnionFileSystem::Mount)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkUnionFileSystem::Mount)
    HK_TRACKER_MEMBER(hkUnionFileSystem::Mount, m_fs, 0, "hkFileSystem *") // class hkFileSystem *
    HK_TRACKER_MEMBER(hkUnionFileSystem::Mount, m_srcPath, 0, "hkStringPtr") // hkStringPtr
    HK_TRACKER_MEMBER(hkUnionFileSystem::Mount, m_dstPath, 0, "hkStringPtr") // hkStringPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkUnionFileSystem::Mount, s_libraryName)

#include <Common/Base/System/Io/FileSystem/hkFileSystem.h>


// hkFileSystem ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkFileSystem)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(TimeStamp)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Entry)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Iterator)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(DirectoryListing)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(AccessMode)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Result)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(OpenFlags)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkFileSystem)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkFileSystem, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkFileSystem, AccessMode, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkFileSystem, Result, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkFileSystem, OpenFlags, s_libraryName)


// TimeStamp hkFileSystem
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkFileSystem, TimeStamp, s_libraryName)


// Entry hkFileSystem

HK_TRACKER_DECLARE_CLASS_BEGIN(hkFileSystem::Entry)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(FlagValues)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkFileSystem::Entry)
    HK_TRACKER_MEMBER(hkFileSystem::Entry, m_fs, 0, "hkFileSystem*") // class hkFileSystem*
    HK_TRACKER_MEMBER(hkFileSystem::Entry, m_path, 0, "hkStringPtr") // hkStringPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkFileSystem::Entry, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkFileSystem::Entry, FlagValues, s_libraryName)


// Iterator hkFileSystem

HK_TRACKER_DECLARE_CLASS_BEGIN(hkFileSystem::Iterator)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Impl)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkFileSystem::Iterator)
    HK_TRACKER_MEMBER(hkFileSystem::Iterator, m_fs, 0, "hkFileSystem *") // class hkFileSystem *
    HK_TRACKER_MEMBER(hkFileSystem::Iterator, m_wildcard, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkFileSystem::Iterator, m_impl, 0, "hkFileSystem::Iterator::Impl *") // struct hkFileSystem::Iterator::Impl *
    HK_TRACKER_MEMBER(hkFileSystem::Iterator, m_entry, 0, "hkFileSystem::Entry") // struct hkFileSystem::Entry
    HK_TRACKER_MEMBER(hkFileSystem::Iterator, m_todo, 0, "hkArray<hkStringPtr, hkContainerHeapAllocator>") // hkArray< hkStringPtr, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkFileSystem::Iterator, s_libraryName)


// Impl hkFileSystem::Iterator

HK_TRACKER_DECLARE_CLASS_BEGIN(hkFileSystem::Iterator::Impl)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkFileSystem::Iterator::Impl)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkFileSystem::Iterator::Impl, s_libraryName, hkReferencedObject)


// DirectoryListing hkFileSystem

HK_TRACKER_DECLARE_CLASS_BEGIN(hkFileSystem::DirectoryListing)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkFileSystem::DirectoryListing)
    HK_TRACKER_MEMBER(hkFileSystem::DirectoryListing, m_entries, 0, "hkArray<hkFileSystem::Entry, hkContainerHeapAllocator>") // hkArray< struct hkFileSystem::Entry, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkFileSystem::DirectoryListing, m_fs, 0, "hkFileSystem*") // class hkFileSystem*
    HK_TRACKER_MEMBER(hkFileSystem::DirectoryListing, m_top, 0, "hkStringPtr") // hkStringPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkFileSystem::DirectoryListing, s_libraryName)

#include <Common/Base/System/Io/FileSystem/hkServerFileSystem.h>


// hkServerFileSystem ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkServerFileSystem)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Mode)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Version)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(OutCommands)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(InCommands)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkServerFileSystem)
    HK_TRACKER_MEMBER(hkServerFileSystem, m_listenSocket, 0, "hkSocket*") // class hkSocket*
    HK_TRACKER_MEMBER(hkServerFileSystem, m_connectSocket, 0, "hkSocket*") // class hkSocket*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkServerFileSystem, s_libraryName, hkWin32FileSystem)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkServerFileSystem, Mode, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkServerFileSystem, Version, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkServerFileSystem, OutCommands, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkServerFileSystem, InCommands, s_libraryName)

#include <Common/Base/System/Io/IArchive/hkIArchive.h>


// hkIArchive ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkIArchive)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkIArchive)
    HK_TRACKER_MEMBER(hkIArchive, m_streamReader, 0, "hkStreamReader *") // class hkStreamReader *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkIArchive, s_libraryName, hkReferencedObject)

#include <Common/Base/System/Io/IStream/hkIStream.h>


// hkIstream ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkIstream)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkIstream)
    HK_TRACKER_MEMBER(hkIstream, m_streamReader, 0, "hkStreamReader *") // class hkStreamReader *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkIstream, s_libraryName, hkReferencedObject)

#include <Common/Base/System/Io/OArchive/hkOArchive.h>


// hkOArchive ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkOArchive)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkOArchive)
    HK_TRACKER_MEMBER(hkOArchive, m_writer, 0, "hkStreamWriter *") // class hkStreamWriter *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkOArchive, s_libraryName, hkReferencedObject)

#include <Common/Base/System/Io/OStream/hkOStream.h>


// hkOstream ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkOstream)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CustomFormater)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkOstream)
    HK_TRACKER_MEMBER(hkOstream, m_writer, 0, "hkStreamWriter *") // class hkStreamWriter *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkOstream, s_libraryName, hkReferencedObject)


// CustomFormater hkOstream

HK_TRACKER_DECLARE_CLASS_BEGIN(hkOstream::CustomFormater)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkOstream::CustomFormater)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkOstream::CustomFormater, s_libraryName)

#include <Common/Base/System/Io/Reader/Buffered/hkBufferedStreamReader.h>


// hkBufferedStreamReader ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkBufferedStreamReader)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Buffer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkBufferedStreamReader)
    HK_TRACKER_MEMBER(hkBufferedStreamReader, m_stream, 0, "hkStreamReader*") // class hkStreamReader*
    HK_TRACKER_MEMBER(hkBufferedStreamReader, m_seekStream, 0, "hkSeekableStreamReader*") // class hkSeekableStreamReader*
    HK_TRACKER_MEMBER(hkBufferedStreamReader, m_buf, 0, "hkBufferedStreamReader::Buffer") // struct hkBufferedStreamReader::Buffer
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkBufferedStreamReader, s_libraryName, hkSeekableStreamReader)


// Buffer hkBufferedStreamReader

HK_TRACKER_DECLARE_CLASS_BEGIN(hkBufferedStreamReader::Buffer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkBufferedStreamReader::Buffer)
    HK_TRACKER_MEMBER(hkBufferedStreamReader::Buffer, begin, 0, "char*") // char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkBufferedStreamReader::Buffer, s_libraryName)

#include <Common/Base/System/Io/Reader/Compressed/hkCompressedStreamReader.h>


// hkCompressedStreamReader ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkCompressedStreamReader)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkCompressedStreamReader)
    HK_TRACKER_MEMBER(hkCompressedStreamReader, m_stream, 0, "hkStreamReader*") // class hkStreamReader*
    HK_TRACKER_MEMBER(hkCompressedStreamReader, m_compbuf, 0, "hkUint8*") // hkUint8*
    HK_TRACKER_MEMBER(hkCompressedStreamReader, m_uncompbuf, 0, "hkUint8*") // hkUint8*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkCompressedStreamReader, s_libraryName, hkStreamReader)

#include <Common/Base/System/Io/Reader/FileServer/hkFileServerStreamReader.h>


// hkFileServerStreamReader ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkFileServerStreamReader)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(OutCommands)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(InCommands)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkFileServerStreamReader)
    HK_TRACKER_MEMBER(hkFileServerStreamReader, m_socket, 0, "hkSocket*") // class hkSocket*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkFileServerStreamReader, s_libraryName, hkSeekableStreamReader)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkFileServerStreamReader, OutCommands, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkFileServerStreamReader, InCommands, s_libraryName)

#include <Common/Base/System/Io/Reader/Memory/hkMemoryStreamReader.h>


// hkMemoryStreamReader ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMemoryStreamReader)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MemoryType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMemoryStreamReader)
    HK_TRACKER_MEMBER(hkMemoryStreamReader, m_buf, 0, "char*") // char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkMemoryStreamReader, s_libraryName, hkSeekableStreamReader)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkMemoryStreamReader, MemoryType, s_libraryName)

#include <Common/Base/System/Io/Reader/hkStreamReader.h>


// hkStreamReader ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkStreamReader)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkStreamReader)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkStreamReader, s_libraryName, hkReferencedObject)


// hkSeekableStreamReader ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkSeekableStreamReader)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SeekWhence)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkSeekableStreamReader)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkSeekableStreamReader, s_libraryName, hkStreamReader)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkSeekableStreamReader, SeekWhence, s_libraryName)

#include <Common/Base/System/Io/Socket/hkSocket.h>


// hkSocket ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkSocket)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ReaderAdapter)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(WriterAdapter)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SOCKET_EVENTS)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkSocket)
    HK_TRACKER_MEMBER(hkSocket, m_reader, 0, "hkSocket::ReaderAdapter") // class hkSocket::ReaderAdapter
    HK_TRACKER_MEMBER(hkSocket, m_writer, 0, "hkSocket::WriterAdapter") // class hkSocket::WriterAdapter
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkSocket, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkSocket, SOCKET_EVENTS, s_libraryName)


// ReaderAdapter hkSocket

HK_TRACKER_DECLARE_CLASS_BEGIN(hkSocket::ReaderAdapter)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkSocket::ReaderAdapter)
    HK_TRACKER_MEMBER(hkSocket::ReaderAdapter, m_socket, 0, "hkSocket*") // class hkSocket*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkSocket::ReaderAdapter, s_libraryName, hkStreamReader)


// WriterAdapter hkSocket

HK_TRACKER_DECLARE_CLASS_BEGIN(hkSocket::WriterAdapter)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkSocket::WriterAdapter)
    HK_TRACKER_MEMBER(hkSocket::WriterAdapter, m_socket, 0, "hkSocket*") // class hkSocket*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkSocket::WriterAdapter, s_libraryName, hkStreamWriter)

#include <Common/Base/System/Io/Util/hkLoadUtil.h>


// hkLoadUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkLoadUtil)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkLoadUtil)
    HK_TRACKER_MEMBER(hkLoadUtil, m_fileName, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkLoadUtil, m_reader, 0, "hkStreamReader *") // class hkStreamReader *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkLoadUtil, s_libraryName)

#include <Common/Base/System/Io/Writer/Array/hkArrayStreamWriter.h>


// hkMemoryTrack ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMemoryTrack)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMemoryTrack)
    HK_TRACKER_MEMBER(hkMemoryTrack, m_sectors, 0, "hkArray<hkUint8*, hkContainerHeapAllocator>") // hkArray< hkUint8*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkMemoryTrack, s_libraryName)


// hkMemoryTrackStreamWriter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMemoryTrackStreamWriter)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(TrackOwnership)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMemoryTrackStreamWriter)
    HK_TRACKER_MEMBER(hkMemoryTrackStreamWriter, m_track, 0, "hkMemoryTrack*") // class hkMemoryTrack*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkMemoryTrackStreamWriter, s_libraryName, hkStreamWriter)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkMemoryTrackStreamWriter, TrackOwnership, s_libraryName)


// hkMemoryTrackStreamReader ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMemoryTrackStreamReader)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MemoryType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMemoryTrackStreamReader)
    HK_TRACKER_MEMBER(hkMemoryTrackStreamReader, m_track, 0, "hkMemoryTrack*") // const class hkMemoryTrack*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkMemoryTrackStreamReader, s_libraryName, hkStreamReader)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkMemoryTrackStreamReader, MemoryType, s_libraryName)


// hkArrayStreamWriter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkArrayStreamWriter)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ArrayOwnership)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkArrayStreamWriter)
    HK_TRACKER_MEMBER(hkArrayStreamWriter, m_arr, 0, "hkArrayBase<char>*") // class hkArrayBase< char >*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkArrayStreamWriter, s_libraryName, hkStreamWriter)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkArrayStreamWriter, ArrayOwnership, s_libraryName)

#include <Common/Base/System/Io/Writer/Buffered/hkBufferedStreamWriter.h>


// hkBufferedStreamWriter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkBufferedStreamWriter)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkBufferedStreamWriter)
    HK_TRACKER_MEMBER(hkBufferedStreamWriter, m_stream, 0, "hkStreamWriter*") // class hkStreamWriter*
    HK_TRACKER_MEMBER(hkBufferedStreamWriter, m_buf, 0, "char*") // char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkBufferedStreamWriter, s_libraryName, hkStreamWriter)

#include <Common/Base/System/Io/Writer/Compressed/hkCompressedStreamWriter.h>


// hkCompressedStreamWriter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkCompressedStreamWriter)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkCompressedStreamWriter)
    HK_TRACKER_MEMBER(hkCompressedStreamWriter, m_stream, 0, "hkStreamWriter*") // class hkStreamWriter*
    HK_TRACKER_MEMBER(hkCompressedStreamWriter, m_uncompbuf, 0, "hkUint8*") // hkUint8*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkCompressedStreamWriter, s_libraryName, hkStreamWriter)

#include <Common/Base/System/Io/Writer/Crc/hkCrcStreamWriter.h>


// hkCrc32StreamWriter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkCrc32StreamWriter)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkCrc32StreamWriter)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkCrc32StreamWriter, s_libraryName, hkCrcStreamWriterunsignedint3988292384)


// hkCrc64StreamWriter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkCrc64StreamWriter)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkCrc64StreamWriter)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkCrc64StreamWriter, s_libraryName, hkCrcStreamWriterunsignedlonglong14514072000185962306)

#include <Common/Base/System/Io/Writer/FileServer/hkFileServerStreamWriter.h>


// hkFileServerStreamWriter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkFileServerStreamWriter)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(OutCommands)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(InCommands)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkFileServerStreamWriter)
    HK_TRACKER_MEMBER(hkFileServerStreamWriter, m_socket, 0, "hkSocket*") // class hkSocket*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkFileServerStreamWriter, s_libraryName, hkStreamWriter)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkFileServerStreamWriter, OutCommands, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkFileServerStreamWriter, InCommands, s_libraryName)

#include <Common/Base/System/Io/Writer/Memory/hkMemoryStreamWriter.h>


// hkMemoryStreamWriter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMemoryStreamWriter)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMemoryStreamWriter)
    HK_TRACKER_MEMBER(hkMemoryStreamWriter, m_buf, 0, "void*") // void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkMemoryStreamWriter, s_libraryName, hkStreamWriter)

#include <Common/Base/System/Io/Writer/OffsetOnly/hkOffsetOnlyStreamWriter.h>


// hkOffsetOnlyStreamWriter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkOffsetOnlyStreamWriter)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkOffsetOnlyStreamWriter)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkOffsetOnlyStreamWriter, s_libraryName, hkStreamWriter)

#include <Common/Base/System/Io/Writer/Printf/hkPrintfStreamWriter.h>


// hkPrintfStreamWriter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkPrintfStreamWriter)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkPrintfStreamWriter)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkPrintfStreamWriter, s_libraryName, hkStreamWriter)

#include <Common/Base/System/Io/Writer/SubStream/hkSubStreamWriter.h>


// hkSubStreamWriter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkSubStreamWriter)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkSubStreamWriter)
    HK_TRACKER_MEMBER(hkSubStreamWriter, m_childStream, 0, "hkStreamWriter*") // class hkStreamWriter*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkSubStreamWriter, s_libraryName, hkStreamWriter)

#include <Common/Base/System/Io/Writer/VdbCommand/hkVdbCommandWriter.h>


// hkVdbCommandWriter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkVdbCommandWriter)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkVdbCommandWriter)
    HK_TRACKER_MEMBER(hkVdbCommandWriter, m_buffer, 0, "hkArray<char, hkContainerDebugAllocator>") // hkArray< char, struct hkContainerDebugAllocator >
    HK_TRACKER_MEMBER(hkVdbCommandWriter, m_writer, 0, "hkStreamWriter*") // class hkStreamWriter*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkVdbCommandWriter, s_libraryName, hkStreamWriter)

#include <Common/Base/System/Io/Writer/hkStreamWriter.h>


// hkStreamWriter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkStreamWriter)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SeekWhence)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkStreamWriter)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkStreamWriter, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkStreamWriter, SeekWhence, s_libraryName)

#include <Common/Base/System/StackTracer/hkStackTracer.h>


// hkStackTracer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkStackTracer)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CallTree)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkStackTracer)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkStackTracer, s_libraryName)


// CallTree hkStackTracer

HK_TRACKER_DECLARE_CLASS_BEGIN(hkStackTracer::CallTree)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Node)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkStackTracer::CallTree)
    HK_TRACKER_MEMBER(hkStackTracer::CallTree, m_nodes, 0, "hkArrayBase<hkStackTracer::CallTree::Node>") // class hkArrayBase< struct hkStackTracer::CallTree::Node >
    HK_TRACKER_MEMBER(hkStackTracer::CallTree, m_allocator, 0, "hkMemoryAllocator*") // class hkMemoryAllocator*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkStackTracer::CallTree, s_libraryName)


// Node hkStackTracer::CallTree
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkStackTracer::CallTree, Node, s_libraryName)

#include <Common/Base/System/Stopwatch/hkStopwatch.h>


// hkStopwatch ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkStopwatch)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkStopwatch)
    HK_TRACKER_MEMBER(hkStopwatch, m_name, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkStopwatch, s_libraryName)

#include <Common/Base/System/hkBaseSystem.h>


// hkHardwareInfo ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkHardwareInfo, s_libraryName)

#include <Common/Base/Thread/CpuCache/hkCpuCache.h>


// hkCpuCache ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkCpuCache)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Cinfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkCpuCache)
    HK_TRACKER_MEMBER(hkCpuCache, m_names, 0, "char* [32]") // const char* [32]
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkCpuCache, s_libraryName)


// Cinfo hkCpuCache

HK_TRACKER_DECLARE_CLASS_BEGIN(hkCpuCache::Cinfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkCpuCache::Cinfo)
    HK_TRACKER_MEMBER(hkCpuCache::Cinfo, m_memoryStart, 0, "void*") // void*
    HK_TRACKER_MEMBER(hkCpuCache::Cinfo, m_workingBuffer, 0, "void*") // void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkCpuCache::Cinfo, s_libraryName)

#include <Common/Base/Thread/CriticalSection/hkCriticalSection.h>


// hkCriticalSection ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkCriticalSection, s_libraryName)


// hkCriticalSectionLock ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkCriticalSectionLock)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkCriticalSectionLock)
    HK_TRACKER_MEMBER(hkCriticalSectionLock, m_section, 0, "hkCriticalSection*") // class hkCriticalSection*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkCriticalSectionLock, s_libraryName)

#include <Common/Base/Thread/Job/ThreadPool/Cpu/hkCpuJobThreadPool.h>


// hkCpuJobThreadPoolCinfo ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkCpuJobThreadPoolCinfo, s_libraryName)


// hkCpuJobThreadPool ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkCpuJobThreadPool)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkCpuJobThreadPool)
    HK_TRACKER_MEMBER(hkCpuJobThreadPool, m_threadPool, 0, "hkCpuThreadPool") // class hkCpuThreadPool
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkCpuJobThreadPool, s_libraryName, hkJobThreadPool)

#include <Common/Base/Thread/Job/ThreadPool/hkJobThreadPool.h>


// hkJobThreadPool ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkJobThreadPool)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkJobThreadPool)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkJobThreadPool, s_libraryName, hkThreadPool)

#include <Common/Base/Thread/JobQueue/hkJobQueue.h>


// hkJob ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkJob, s_libraryName)


// hkExternalJobProfiler ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkExternalJobProfiler)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkExternalJobProfiler)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkExternalJobProfiler, s_libraryName)


// hkJobQueueHwSetup ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkJobQueueHwSetup)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CellRules)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SpuSchedulePolicy)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkJobQueueHwSetup)
    HK_TRACKER_MEMBER(hkJobQueueHwSetup, m_threadIdsSharingCaches, 0, "hkArray<hkArray<hkInt32, hkContainerHeapAllocator>, hkContainerHeapAllocator>") // hkArray< hkArray< hkInt32, struct hkContainerHeapAllocator >, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkJobQueueHwSetup, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkJobQueueHwSetup, CellRules, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkJobQueueHwSetup, SpuSchedulePolicy, s_libraryName)


// hkJobQueueCinfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkJobQueueCinfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkJobQueueCinfo)
    HK_TRACKER_MEMBER(hkJobQueueCinfo, m_jobQueueHwSetup, 0, "hkJobQueueHwSetup") // struct hkJobQueueHwSetup
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkJobQueueCinfo, s_libraryName)


// hkJobQueue ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkJobQueue)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(JobQueueEntry)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(JobTypeSpuStats)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(DynamicData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(JobQueueEntryInput)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(hkJobHandlerFuncs)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CustomJobType)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CustomJobTypeSetup)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(WaitPolicy)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(JobPriority)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(JobPopFuncResult)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(JobCreationStatus)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(WaitStatus)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(JobStatus)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(FinishJobFlag)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkJobQueue)
    HK_TRACKER_MEMBER(hkJobQueue, m_data, 0, "hkJobQueue::DynamicData*") // struct hkJobQueue::DynamicData*
    HK_TRACKER_MEMBER(hkJobQueue, m_hwSetup, 0, "hkJobQueueHwSetup") // struct hkJobQueueHwSetup
    HK_TRACKER_MEMBER(hkJobQueue, m_queueSemaphores, 0, "hkSemaphore* [5]") // class hkSemaphore* [5]
    HK_TRACKER_MEMBER(hkJobQueue, m_jobFuncs, 0, "hkJobQueue::hkJobHandlerFuncs [20]") // struct hkJobQueue::hkJobHandlerFuncs [20]
    HK_TRACKER_MEMBER(hkJobQueue, m_threadPool, 0, "hkSpuJobThreadPool*") // class hkSpuJobThreadPool*
    HK_TRACKER_MEMBER(hkJobQueue, m_customJobSetup, 0, "hkArray<hkJobQueue::CustomJobTypeSetup, hkContainerHeapAllocator>") // hkArray< struct hkJobQueue::CustomJobTypeSetup, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkJobQueue, m_externalJobProfiler, 0, "hkExternalJobProfiler*") // class hkExternalJobProfiler*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkJobQueue, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkJobQueue, WaitPolicy, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkJobQueue, JobPriority, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkJobQueue, JobPopFuncResult, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkJobQueue, JobCreationStatus, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkJobQueue, WaitStatus, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkJobQueue, JobStatus, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkJobQueue, FinishJobFlag, s_libraryName)


// JobQueueEntry hkJobQueue
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkJobQueue, JobQueueEntry, s_libraryName)


// JobTypeSpuStats hkJobQueue
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkJobQueue, JobTypeSpuStats, s_libraryName)


// DynamicData hkJobQueue

HK_TRACKER_DECLARE_CLASS_BEGIN(hkJobQueue::DynamicData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkJobQueue::DynamicData)
    HK_TRACKER_MEMBER(hkJobQueue::DynamicData, m_jobQueue, 0, "hkQueue<hkJobQueue::JobQueueEntry> [25]") // class hkQueue< struct hkJobQueue::JobQueueEntry > [25]
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkJobQueue::DynamicData, s_libraryName)


// JobQueueEntryInput hkJobQueue
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkJobQueue, JobQueueEntryInput, s_libraryName)


// hkJobHandlerFuncs hkJobQueue
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkJobQueue, hkJobHandlerFuncs, s_libraryName)


// CustomJobType hkJobQueue
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkJobQueue, CustomJobType, s_libraryName)


// CustomJobTypeSetup hkJobQueue
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkJobQueue, CustomJobTypeSetup, s_libraryName)

// None hkJobType
HK_TRACKER_IMPLEMENT_SIMPLE(hkJobType, s_libraryName)
// None hkJobSpuType
HK_TRACKER_IMPLEMENT_SIMPLE(hkJobSpuType, s_libraryName)
#include <Common/Base/Thread/Pool/hkCpuThreadPool.h>


// hkCpuThreadPool ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkCpuThreadPool)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(WorkerThreadData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SharedThreadData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkCpuThreadPool)
    HK_TRACKER_MEMBER(hkCpuThreadPool, m_workerThreads, 0, "hkCpuThreadPool::WorkerThreadData [12]") // struct hkCpuThreadPool::WorkerThreadData [12]
    HK_TRACKER_MEMBER(hkCpuThreadPool, m_sharedThreadData, 0, "hkCpuThreadPool::SharedThreadData") // struct hkCpuThreadPool::SharedThreadData
    HK_TRACKER_MEMBER(hkCpuThreadPool, m_threadName, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkCpuThreadPool, s_libraryName, hkThreadPool)


// WorkerThreadData hkCpuThreadPool

HK_TRACKER_DECLARE_CLASS_BEGIN(hkCpuThreadPool::WorkerThreadData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkCpuThreadPool::WorkerThreadData)
    HK_TRACKER_MEMBER(hkCpuThreadPool::WorkerThreadData, m_sharedThreadData, 0, "hkCpuThreadPool::SharedThreadData*") // struct hkCpuThreadPool::SharedThreadData*
    HK_TRACKER_MEMBER(hkCpuThreadPool::WorkerThreadData, m_thread, 0, "hkThread") // class hkThread
    HK_TRACKER_MEMBER(hkCpuThreadPool::WorkerThreadData, m_semaphore, 0, "hkSemaphore") // class hkSemaphore
    HK_TRACKER_MEMBER(hkCpuThreadPool::WorkerThreadData, m_monitorStreamBegin, 0, "char*") // char*
    HK_TRACKER_MEMBER(hkCpuThreadPool::WorkerThreadData, m_monitorStreamEnd, 0, "char*") // char*
    HK_TRACKER_MEMBER(hkCpuThreadPool::WorkerThreadData, m_context, 0, "hkWorkerThreadContext*") // class hkWorkerThreadContext*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkCpuThreadPool::WorkerThreadData, s_libraryName)


// SharedThreadData hkCpuThreadPool

HK_TRACKER_DECLARE_CLASS_BEGIN(hkCpuThreadPool::SharedThreadData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkCpuThreadPool::SharedThreadData)
    HK_TRACKER_MEMBER(hkCpuThreadPool::SharedThreadData, m_workLoad, 0, "void*") // void*
    HK_TRACKER_MEMBER(hkCpuThreadPool::SharedThreadData, m_workerThreadFinished, 0, "hkSemaphore") // class hkSemaphore
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkCpuThreadPool::SharedThreadData, s_libraryName)


// hkCpuThreadPoolCinfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkCpuThreadPoolCinfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkCpuThreadPoolCinfo)
    HK_TRACKER_MEMBER(hkCpuThreadPoolCinfo, m_hardwareThreadIds, 0, "hkArray<hkInt32, hkContainerHeapAllocator>") // hkArray< hkInt32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkCpuThreadPoolCinfo, m_threadName, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkCpuThreadPoolCinfo, s_libraryName)

#include <Common/Base/Thread/Pool/hkThreadPool.h>


// hkThreadPool ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkThreadPool)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkThreadPool)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkThreadPool, s_libraryName, hkReferencedObject)

#include <Common/Base/Thread/Semaphore/hkSemaphore.h>


// hkSemaphore ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkSemaphore)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkSemaphore)
    HK_TRACKER_MEMBER(hkSemaphore, m_semaphore, 0, "void*") // void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkSemaphore, s_libraryName)

#include <Common/Base/Thread/SimpleScheduler/hkSimpleScheduler.h>


// hkSimpleSchedulerTaskBuilder ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkSimpleSchedulerTaskBuilder)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(TaskIdDiscriminant)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Object)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(TaskHeader)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(TaskInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkSimpleSchedulerTaskBuilder)
    HK_TRACKER_MEMBER(hkSimpleSchedulerTaskBuilder, m_initialReferenceCounts, 0, "hkArray<hkSimpleSchedulerTaskBuilder::TaskInfo, hkContainerHeapAllocator>") // hkArray< struct hkSimpleSchedulerTaskBuilder::TaskInfo, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkSimpleSchedulerTaskBuilder, m_taskData, 0, "hkArray<hkUint8, hkContainerHeapAllocator>") // hkArray< hkUint8, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkSimpleSchedulerTaskBuilder, m_finishedTasks, 0, "hkArray<hkHandle<hkUint16, 65535, hkSimpleSchedulerTaskBuilder::TaskIdDiscriminant>, hkContainerHeapAllocator>") // hkArray< struct hkHandle< hkUint16, 65535, struct hkSimpleSchedulerTaskBuilder::TaskIdDiscriminant >, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkSimpleSchedulerTaskBuilder, s_libraryName)


// TaskIdDiscriminant hkSimpleSchedulerTaskBuilder
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkSimpleSchedulerTaskBuilder, TaskIdDiscriminant, s_libraryName)


// Object hkSimpleSchedulerTaskBuilder
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkSimpleSchedulerTaskBuilder, Object, s_libraryName)


// TaskHeader hkSimpleSchedulerTaskBuilder
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkSimpleSchedulerTaskBuilder, TaskHeader, s_libraryName)


// TaskInfo hkSimpleSchedulerTaskBuilder
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkSimpleSchedulerTaskBuilder, TaskInfo, s_libraryName)


// hkSimpleScheduler ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkSimpleScheduler)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkSimpleScheduler)
    HK_TRACKER_MEMBER(hkSimpleScheduler, m_referenceCounts, 0, "hkArray<hkSimpleSchedulerTaskBuilder::TaskInfo, hkContainerHeapAllocator>") // hkArray< struct hkSimpleSchedulerTaskBuilder::TaskInfo, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkSimpleScheduler, m_finishedTasks, 0, "hkUint16*") // hkUint16*
    HK_TRACKER_MEMBER(hkSimpleScheduler, m_numFinishedTasks, 0, "hkUint32*") // hkUint32*
    HK_TRACKER_MEMBER(hkSimpleScheduler, m_taskData, 0, "hkUint8*") // const hkUint8*
    HK_TRACKER_MEMBER(hkSimpleScheduler, m_schedulerPpu, 0, "hkSimpleSchedulerTaskBuilder*") // const class hkSimpleSchedulerTaskBuilder*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkSimpleScheduler, s_libraryName)

#include <Common/Base/Thread/Task/hkTask.h>


// hkTask ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkTask)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkTask)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkTask, s_libraryName, hkReferencedObject)


// hkTaskType ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkTaskType, s_libraryName)

#include <Common/Base/Thread/Task/hkTaskGraph.h>


// hkTaskGraph ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkTaskGraph)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Dependency)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkTaskGraph)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkTaskGraph, s_libraryName)


// Dependency hkTaskGraph
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkTaskGraph, Dependency, s_libraryName)


// hkDefaultTaskGraph ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDefaultTaskGraph)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(TaskIdDiscriminant)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(TaskInfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(TaskDetph)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ExecutionPolicy)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDefaultTaskGraph)
    HK_TRACKER_MEMBER(hkDefaultTaskGraph, m_taskInfos, 0, "hkArray<hkDefaultTaskGraph::TaskInfo, hkContainerHeapAllocator>") // hkArray< struct hkDefaultTaskGraph::TaskInfo, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkDefaultTaskGraph, m_children, 0, "hkArray<hkHandle<hkUint16, 65535, hkDefaultTaskGraph::TaskIdDiscriminant>, hkContainerHeapAllocator>") // hkArray< struct hkHandle< hkUint16, 65535, struct hkDefaultTaskGraph::TaskIdDiscriminant >, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkDefaultTaskGraph, m_dependencies, 0, "hkArray<hkTaskGraph::Dependency, hkContainerHeapAllocator>") // hkArray< struct hkTaskGraph::Dependency, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkDefaultTaskGraph, s_libraryName, hkTaskGraph)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkDefaultTaskGraph, ExecutionPolicy, s_libraryName)


// TaskIdDiscriminant hkDefaultTaskGraph
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkDefaultTaskGraph, TaskIdDiscriminant, s_libraryName)


// TaskInfo hkDefaultTaskGraph

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDefaultTaskGraph::TaskInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDefaultTaskGraph::TaskInfo)
    HK_TRACKER_MEMBER(hkDefaultTaskGraph::TaskInfo, m_task, 0, "hkTask*") // class hkTask*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkDefaultTaskGraph::TaskInfo, s_libraryName)


// TaskDetph hkDefaultTaskGraph
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkDefaultTaskGraph, TaskDetph, s_libraryName)

#include <Common/Base/Thread/Task/hkTaskGraphUtil.h>


// hkTaskGraphUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkTaskGraphUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(TaskPrinter)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkTaskGraphUtil, s_libraryName)


// TaskPrinter hkTaskGraphUtil

HK_TRACKER_DECLARE_CLASS_BEGIN(hkTaskGraphUtil::TaskPrinter)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkTaskGraphUtil::TaskPrinter)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkTaskGraphUtil::TaskPrinter, s_libraryName)

#include <Common/Base/Thread/Task/hkTaskQueue.h>


// hkTaskQueue ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkTaskQueue)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(GraphIdDiscriminant)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(PrioritizedTask)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(GraphInfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(WaitingMode)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(GetNextTaskResult)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(LockMode)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(TaskType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkTaskQueue)
    HK_TRACKER_MEMBER(hkTaskQueue, m_graphInfos, 0, "hkFreeListArray<hkTaskQueue::GraphInfo, hkHandle<hkUint8, 255, hkTaskQueue::GraphIdDiscriminant>, 8, hkTaskQueue::GraphInfo::FreeListArrayOperations>") // struct hkFreeListArray< struct hkTaskQueue::GraphInfo, struct hkHandle< hkUint8, 255, struct hkTaskQueue::GraphIdDiscriminant >, 8, struct hkTaskQueue::GraphInfo::FreeListArrayOperations >
    HK_TRACKER_MEMBER(hkTaskQueue, m_graphSignals, 0, "hkSemaphore [8]") // class hkSemaphore [8]
    HK_TRACKER_MEMBER(hkTaskQueue, m_queue, 0, "hkMinHeap<hkTaskQueue::PrioritizedTask, hkMinHeapDefaultOperations<hkTaskQueue::PrioritizedTask> >") // class hkMinHeap< struct hkTaskQueue::PrioritizedTask, struct hkMinHeapDefaultOperations< struct hkTaskQueue::PrioritizedTask > >
    HK_TRACKER_MEMBER(hkTaskQueue, m_queueSpu, 0, "hkMinHeap<hkTaskQueue::PrioritizedTask, hkMinHeapDefaultOperations<hkTaskQueue::PrioritizedTask> >") // class hkMinHeap< struct hkTaskQueue::PrioritizedTask, struct hkMinHeapDefaultOperations< struct hkTaskQueue::PrioritizedTask > >
    HK_TRACKER_MEMBER(hkTaskQueue, m_taskAvailableSignal, 0, "hkSemaphore") // class hkSemaphore
    HK_TRACKER_MEMBER(hkTaskQueue, m_taskAvailableSignalSpu, 0, "hkSemaphore") // class hkSemaphore
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkTaskQueue, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkTaskQueue, WaitingMode, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkTaskQueue, GetNextTaskResult, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkTaskQueue, LockMode, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkTaskQueue, TaskType, s_libraryName)


// GraphIdDiscriminant hkTaskQueue
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkTaskQueue, GraphIdDiscriminant, s_libraryName)


// PrioritizedTask hkTaskQueue
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkTaskQueue, PrioritizedTask, s_libraryName)


// GraphInfo hkTaskQueue

HK_TRACKER_DECLARE_CLASS_BEGIN(hkTaskQueue::GraphInfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(FreeListArrayOperations)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkTaskQueue::GraphInfo)
    HK_TRACKER_MEMBER(hkTaskQueue::GraphInfo, m_scheduler, 0, "hkTaskScheduler") // class hkTaskScheduler
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkTaskQueue::GraphInfo, s_libraryName)


// FreeListArrayOperations hkTaskQueue::GraphInfo
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkTaskQueue::GraphInfo, FreeListArrayOperations, s_libraryName)

#include <Common/Base/Thread/Task/hkTaskScheduler.h>


// hkTaskScheduler ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkTaskScheduler)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(TaskState)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkTaskScheduler)
    HK_TRACKER_MEMBER(hkTaskScheduler, m_taskGraph, 0, "hkDefaultTaskGraph*") // const struct hkDefaultTaskGraph*
    HK_TRACKER_MEMBER(hkTaskScheduler, m_taskStates, 0, "hkArray<hkTaskScheduler::TaskState, hkContainerHeapAllocator>") // hkArray< struct hkTaskScheduler::TaskState, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkTaskScheduler, m_availableTasks, 0, "hkArray<hkHandle<hkUint16, 65535, hkDefaultTaskGraph::TaskIdDiscriminant>, hkContainerHeapAllocator>") // hkArray< struct hkHandle< hkUint16, 65535, struct hkDefaultTaskGraph::TaskIdDiscriminant >, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkTaskScheduler, m_activatedTasks, 0, "hkArray<hkTask*, hkContainerHeapAllocator>") // hkArray< class hkTask*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkTaskScheduler, s_libraryName)


// TaskState hkTaskScheduler
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkTaskScheduler, TaskState, s_libraryName)

#include <Common/Base/Thread/Thread/hkThread.h>


// hkThread ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkThread)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Status)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkThread)
    HK_TRACKER_MEMBER(hkThread, m_thread, 0, "void*") // void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkThread, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkThread, Status, s_libraryName)

#include <Common/Base/Thread/Thread/hkWorkerThreadContext.h>


// hkWorkerThreadContext ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkWorkerThreadContext)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkWorkerThreadContext)
    HK_TRACKER_MEMBER(hkWorkerThreadContext, m_memoryRouter, 0, "hkMemoryRouter") // class hkMemoryRouter
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkWorkerThreadContext, s_libraryName)

#include <Common/Base/Types/Color/hkColor.h>


// hkColor ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkColor, s_libraryName)

#include <Common/Base/Types/Geometry/Aabb/hkAabb.h>


// hkAabb ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkAabb, s_libraryName)


// hkAabbUint32 ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkAabbUint32, s_libraryName)

#include <Common/Base/Types/Geometry/Aabb/hkAabbHalf.h>


// hkAabbHalf ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkAabbHalf, s_libraryName)

#include <Common/Base/Types/Geometry/Aabb/hkAabbUtil.h>


// OffsetAabbInput hkAabbUtil

HK_TRACKER_DECLARE_CLASS_BEGIN(hkAabbUtil::OffsetAabbInput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkAabbUtil::OffsetAabbInput)
    HK_TRACKER_MEMBER(hkAabbUtil::OffsetAabbInput, m_motionState, 0, "hkPadSpu<hkMotionState*>") // class hkPadSpu< const class hkMotionState* >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkAabbUtil::OffsetAabbInput, s_libraryName)

#include <Common/Base/Types/Geometry/Aabb16/hkAabb16.h>


// hkAabb16 ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkAabb16, s_libraryName)

#include <Common/Base/Types/Geometry/Geometry/hkGeometryUtil.h>


// GridInput hkGeometryUtil
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkGeometryUtil::GridInput, s_libraryName, hkGeometryUtil_GridInput)

#include <Common/Base/Types/Geometry/IntSpaceUtil/hkIntSpaceUtil.h>


// hkIntSpaceUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkIntSpaceUtil, s_libraryName)

#include <Common/Base/Types/Geometry/LocalFrame/hkLocalFrame.h>


// hkLocalFrame ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkLocalFrame)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkLocalFrame)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkLocalFrame, s_libraryName, hkReferencedObject)


// hkLocalFrameCollector ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkLocalFrameCollector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkLocalFrameCollector)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkLocalFrameCollector, s_libraryName, hkReferencedObject)


// hkLocalFrameGroup ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkLocalFrameGroup)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkLocalFrameGroup)
    HK_TRACKER_MEMBER(hkLocalFrameGroup, m_name, 0, "hkStringPtr") // hkStringPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkLocalFrameGroup, s_libraryName, hkReferencedObject)


// hkSimpleLocalFrame ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkSimpleLocalFrame)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkSimpleLocalFrame)
    HK_TRACKER_MEMBER(hkSimpleLocalFrame, m_children, 0, "hkArray<hkLocalFrame*, hkContainerHeapAllocator>") // hkArray< class hkLocalFrame*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkSimpleLocalFrame, m_parentFrame, 0, "hkLocalFrame*") // const class hkLocalFrame*
    HK_TRACKER_MEMBER(hkSimpleLocalFrame, m_group, 0, "hkLocalFrameGroup*") // const class hkLocalFrameGroup*
    HK_TRACKER_MEMBER(hkSimpleLocalFrame, m_name, 0, "hkStringPtr") // hkStringPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkSimpleLocalFrame, s_libraryName, hkLocalFrame)

#include <Common/Base/Types/Geometry/Sphere/hkSphere.h>


// hkSphere ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkSphere, s_libraryName)

#include <Common/Base/Types/Geometry/hkGeometry.h>


// hkGeometry ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkGeometry)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Triangle)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(GeometryType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkGeometry)
    HK_TRACKER_MEMBER(hkGeometry, m_vertices, 0, "hkArray<hkVector4f, hkContainerHeapAllocator>") // hkArray< hkVector4f, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkGeometry, m_triangles, 0, "hkArray<hkGeometry::Triangle, hkContainerHeapAllocator>") // hkArray< struct hkGeometry::Triangle, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkGeometry, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkGeometry, GeometryType, s_libraryName)


// Triangle hkGeometry
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkGeometry, Triangle, s_libraryName)

#include <Common/Base/Types/Geometry/hkStridedVertices.h>


// hkStridedVertices ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkStridedVertices, s_libraryName)

#include <Common/Base/Types/Physics/ContactPoint/hkContactPoint.h>


// hkContactPoint ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkContactPoint, s_libraryName)

#include <Common/Base/Types/Physics/ContactPoint/hkContactPointMaterial.h>


// hkContactPointMaterial ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkContactPointMaterial, s_libraryName)

#include <Common/Base/Types/Physics/MotionState/hkMotionState.h>


// hkMotionState ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkMotionState, s_libraryName)

#include <Common/Base/Types/Physics/hkStepInfo.h>


// hkStepInfo ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkStepInfo, s_libraryName)

#include <Common/Base/Types/Properties/hkRefCountedProperties.h>


// hkRefCountedProperties ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkRefCountedProperties)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Entry)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ReferenceCountHandling)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkRefCountedProperties)
    HK_TRACKER_MEMBER(hkRefCountedProperties, m_entries, 0, "hkArray<hkRefCountedProperties::Entry, hkContainerHeapAllocator>") // hkArray< struct hkRefCountedProperties::Entry, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkRefCountedProperties, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkRefCountedProperties, ReferenceCountHandling, s_libraryName)


// Entry hkRefCountedProperties

HK_TRACKER_DECLARE_CLASS_BEGIN(hkRefCountedProperties::Entry)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkRefCountedProperties::Entry)
    HK_TRACKER_MEMBER(hkRefCountedProperties::Entry, m_object, 0, "hkReferencedObject *") // class hkReferencedObject *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkRefCountedProperties::Entry, s_libraryName)

#include <Common/Base/Types/Properties/hkSimpleProperty.h>


// hkSimplePropertyValue ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkSimplePropertyValue, s_libraryName)


// hkSimpleProperty ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkSimpleProperty, s_libraryName)

#include <Common/Base/Types/Traits/hkTraitArithmetic.h>

 // Skipping Class MatchingIntType< 1, 1 > as it is a template

 // Skipping Class MatchingIntType< 1, 0 > as it is a template

 // Skipping Class MatchingIntType< 2, 1 > as it is a template

 // Skipping Class MatchingIntType< 2, 0 > as it is a template

 // Skipping Class MatchingIntType< 4, 1 > as it is a template

 // Skipping Class MatchingIntType< 4, 0 > as it is a template

 // Skipping Class MatchingIntType< 8, 1 > as it is a template

 // Skipping Class MatchingIntType< 8, 0 > as it is a template

 // Skipping Class GreatestPowerOfTwoDivisor< 0 > as it is a template

#include <Common/Base/Types/Traits/hkTraitIsPod.h>

 // Skipping Class IsPodType< * > as it is a template

 // Skipping Class IsPodType<  > as it is a template

 // Skipping Class IsPodType< _Bool > as it is a template

 // Skipping Class IsPodType< hkBool > as it is a template

 // Skipping Class IsPodType< char > as it is a template

 // Skipping Class IsPodType< hkInt8 > as it is a template

 // Skipping Class IsPodType< hkUint8 > as it is a template

 // Skipping Class IsPodType< hkInt16 > as it is a template

 // Skipping Class IsPodType< hkUint16 > as it is a template

 // Skipping Class IsPodType< hkInt32 > as it is a template

 // Skipping Class IsPodType< hkUint32 > as it is a template

 // Skipping Class IsPodType< hkInt64 > as it is a template

 // Skipping Class IsPodType< hkUint64 > as it is a template

 // Skipping Class IsPodType< hkLong > as it is a template

 // Skipping Class IsPodType< hkUlong > as it is a template

 // Skipping Class IsPodType< hkHalf > as it is a template

 // Skipping Class IsPodType< float > as it is a template

 // Skipping Class IsPodType< double > as it is a template

#include <Common/Base/Types/Traits/hkTraitModifier.h>

 // Skipping Class RemoveConst< const  > as it is a template

 // Skipping Class RemoveConst< const * > as it is a template

 // Skipping Class RemoveConst< const & > as it is a template

 // Skipping Class RemoveConst< const const * > as it is a template

 // Skipping Class RemoveConst< const * > as it is a template

 // Skipping Class RemoveRef< & > as it is a template

 // Skipping Class AddConst< const  > as it is a template

 // Skipping Class AddConst< * > as it is a template

 // Skipping Class AddConst< const * > as it is a template

 // Skipping Class AddConst< const * > as it is a template

 // Skipping Class IsConstType< const * > as it is a template

 // Skipping Class IsConstType< const  > as it is a template

 // Skipping Class IsConstType< const * > as it is a template

 // Skipping Class IsConstType< const const * > as it is a template

 // Skipping Class ConstIfTrue< 0,  > as it is a template

#include <Common/Base/Types/hkBaseTypes.h>

 // Skipping Class UnsignedFor< 1 > as it is a template

 // Skipping Class UnsignedFor< 2 > as it is a template

 // Skipping Class UnsignedFor< 4 > as it is a template

 // Skipping Class UnsignedFor< 8 > as it is a template

// hk.MemoryTracker ignore hkFinishLoadedObjectFlag
// hk.MemoryTracker ignore hkVariant
// hk.MemoryTracker ignore hkHalf
// hk.MemoryTracker ignore hkFloat16
// hk.MemoryTracker ignore hkBool
// hk.MemoryTracker ignore hkResult
 // Skipping Class VAR_UNROLL_X_OUT_OF_RANGE< 1 > as it is a template

// hk.MemoryTracker ignore hkCountOfBadArgCheck
// None hkBool32FalseType
HK_TRACKER_IMPLEMENT_SIMPLE(hkBool32FalseType, s_libraryName)
// None hkResultEnum
HK_TRACKER_IMPLEMENT_SIMPLE(hkResultEnum, s_libraryName)
#include <Common/Base/Types/hkRefVariant.h>


// hkRefVariant ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkRefVariant, s_libraryName)

#include <Common/Base/Types/hkSignalSlots.h>

// hk.MemoryTracker ignore hkSlot
// hk.MemoryTracker ignore hkSignal
#include <Common/Base/Types/hkTrait.h>

 // Skipping Class TypesAreEqual< ,  > as it is a template

 // Skipping Class If< 1, ,  > as it is a template

 // Skipping Class If< 0, ,  > as it is a template

#include <Common/Base/Types/hkTypedUnion.h>


// hkTypedUnion ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkTypedUnion)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(EnumVariant)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkTypedUnion)
    HK_TRACKER_MEMBER(hkTypedUnion, m_elem, 0, "hkTypedUnion::Storage") // struct hkTypedUnion::Storage
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkTypedUnion, s_libraryName)


// EnumVariant hkTypedUnion

HK_TRACKER_DECLARE_CLASS_BEGIN(hkTypedUnion::EnumVariant)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkTypedUnion::EnumVariant)
    HK_TRACKER_MEMBER(hkTypedUnion::EnumVariant, m_enum, 0, "hkClassEnum*") // const class hkClassEnum*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkTypedUnion::EnumVariant, s_libraryName)

#include <Common/Base/Types/hkUFloat8.h>

// hk.MemoryTracker ignore hkUFloat8

/*
 * Havok SDK - Base file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */
