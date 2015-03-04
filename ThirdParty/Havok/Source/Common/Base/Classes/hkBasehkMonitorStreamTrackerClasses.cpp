/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#include <Common/Base/hkBase.h>
static const char s_libraryName[] = "hkBasehkMonitorStream";
#include <Common/Base/Memory/Tracker/hkTrackerClassDefinition.h>

void HK_CALL hkBasehkMonitorStreamRegister() {}

#include <Common/Base/Monitor/MonitorStreamAnalyzer/hkMonitorStreamAnalyzer.h>


// hkMonitorStreamStringMap ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMonitorStreamStringMap)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(StringMap)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMonitorStreamStringMap)
    HK_TRACKER_MEMBER(hkMonitorStreamStringMap, m_map, 0, "hkArray<hkMonitorStreamStringMap::StringMap, hkContainerHeapAllocator>") // hkArray< struct hkMonitorStreamStringMap::StringMap, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkMonitorStreamStringMap, s_libraryName)


// StringMap hkMonitorStreamStringMap

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMonitorStreamStringMap::StringMap)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMonitorStreamStringMap::StringMap)
    HK_TRACKER_MEMBER(hkMonitorStreamStringMap::StringMap, m_string, 0, "hkStringPtr") // hkStringPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkMonitorStreamStringMap::StringMap, s_libraryName)


// hkMonitorStreamFrameInfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMonitorStreamFrameInfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(AbsoluteTimeCounter)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMonitorStreamFrameInfo)
    HK_TRACKER_MEMBER(hkMonitorStreamFrameInfo, m_heading, 0, "hkStringPtr") // hkStringPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkMonitorStreamFrameInfo, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkMonitorStreamFrameInfo, AbsoluteTimeCounter, s_libraryName)


// hkMonitorStreamColorTable ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMonitorStreamColorTable)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ColorPair)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMonitorStreamColorTable)
    HK_TRACKER_MEMBER(hkMonitorStreamColorTable, m_colorPairs, 0, "hkArray<hkMonitorStreamColorTable::ColorPair, hkContainerHeapAllocator>") // hkArray< struct hkMonitorStreamColorTable::ColorPair, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkMonitorStreamColorTable, s_libraryName, hkReferencedObject)


// ColorPair hkMonitorStreamColorTable

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMonitorStreamColorTable::ColorPair)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMonitorStreamColorTable::ColorPair)
    HK_TRACKER_MEMBER(hkMonitorStreamColorTable::ColorPair, m_colorName, 0, "hkStringPtr") // hkStringPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkMonitorStreamColorTable::ColorPair, s_libraryName)


// hkMonitorStreamAnalyzer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMonitorStreamAnalyzer)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ThreadDrawInput)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CursorKeys)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CombinedThreadSummaryOptions)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Node)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SampleInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMonitorStreamAnalyzer)
    HK_TRACKER_MEMBER(hkMonitorStreamAnalyzer, m_frameInfos, 0, "hkArray<hkArray<hkMonitorStreamFrameInfo, hkContainerHeapAllocator>, hkContainerHeapAllocator>") // hkArray< hkArray< struct hkMonitorStreamFrameInfo, struct hkContainerHeapAllocator >, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkMonitorStreamAnalyzer, m_data, 0, "hkArray<char, hkContainerDebugAllocator>") // hkArray< char, struct hkContainerDebugAllocator >
    HK_TRACKER_MEMBER(hkMonitorStreamAnalyzer, m_nodeIdForFrameOverview, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkMonitorStreamAnalyzer, s_libraryName)


// ThreadDrawInput hkMonitorStreamAnalyzer

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMonitorStreamAnalyzer::ThreadDrawInput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMonitorStreamAnalyzer::ThreadDrawInput)
    HK_TRACKER_MEMBER(hkMonitorStreamAnalyzer::ThreadDrawInput, m_colorTable, 0, "hkMonitorStreamColorTable*") // struct hkMonitorStreamColorTable*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkMonitorStreamAnalyzer::ThreadDrawInput, s_libraryName)


// CursorKeys hkMonitorStreamAnalyzer
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkMonitorStreamAnalyzer, CursorKeys, s_libraryName)


// CombinedThreadSummaryOptions hkMonitorStreamAnalyzer

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMonitorStreamAnalyzer::CombinedThreadSummaryOptions)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMonitorStreamAnalyzer::CombinedThreadSummaryOptions)
    HK_TRACKER_MEMBER(hkMonitorStreamAnalyzer::CombinedThreadSummaryOptions, m_activeNode, 0, "hkMonitorStreamAnalyzer::Node*") // struct hkMonitorStreamAnalyzer::Node*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkMonitorStreamAnalyzer::CombinedThreadSummaryOptions, s_libraryName)


// Node hkMonitorStreamAnalyzer

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMonitorStreamAnalyzer::Node)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(NodeType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMonitorStreamAnalyzer::Node)
    HK_TRACKER_MEMBER(hkMonitorStreamAnalyzer::Node, m_children, 0, "hkArray<hkMonitorStreamAnalyzer::Node*, hkContainerHeapAllocator>") // hkArray< struct hkMonitorStreamAnalyzer::Node*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkMonitorStreamAnalyzer::Node, m_parent, 0, "hkMonitorStreamAnalyzer::Node*") // struct hkMonitorStreamAnalyzer::Node*
    HK_TRACKER_MEMBER(hkMonitorStreamAnalyzer::Node, m_name, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkMonitorStreamAnalyzer::Node, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkMonitorStreamAnalyzer::Node, NodeType, s_libraryName)


// SampleInfo hkMonitorStreamAnalyzer
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkMonitorStreamAnalyzer, SampleInfo, s_libraryName)

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
