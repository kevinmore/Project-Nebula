/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#include <Physics2012/Internal/hkpInternal.h>
static const char s_libraryName[] = "hkpInternalTreeBroadPhase";
#include <Common/Base/Memory/Tracker/hkTrackerClassDefinition.h>

void HK_CALL hkpInternalTreeBroadPhaseRegister() {}

#include <Physics2012/Internal/BroadPhase/TreeBroadPhase/hkpTreeBroadPhase.h>


// hkpTreeBroadPhase ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpTreeBroadPhase)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Handle)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Group)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(GroupsMasks)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpTreeBroadPhase)
    HK_TRACKER_MEMBER(hkpTreeBroadPhase, m_handles, 0, "hkArray<hkpTreeBroadPhase::Handle, hkContainerHeapAllocator> [2]") // hkArray< struct hkpTreeBroadPhase::Handle, struct hkContainerHeapAllocator > [2]
    HK_TRACKER_MEMBER(hkpTreeBroadPhase, m_childBroadPhase, 0, "hkpBroadPhase*") // class hkpBroadPhase*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpTreeBroadPhase, s_libraryName, hkpBroadPhase)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpTreeBroadPhase, Group, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpTreeBroadPhase, GroupsMasks, s_libraryName)


// Handle hkpTreeBroadPhase

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpTreeBroadPhase::Handle)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpTreeBroadPhase::Handle)
    HK_TRACKER_MEMBER(hkpTreeBroadPhase::Handle, m_bpHandle, 0, "hkpBroadPhaseHandle*") // class hkpBroadPhaseHandle*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpTreeBroadPhase::Handle, s_libraryName)

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
