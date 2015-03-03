/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#include <Geometry/ConvexDecomposition/hkgpConvexDecomposition.h>
static const char s_libraryName[] = "hkgpConvexDecomposition";
#include <Common/Base/Memory/Tracker/hkTrackerClassDefinition.h>

void HK_CALL hkgpConvexDecompositionRegister() {}

#include <Geometry/ConvexDecomposition/hkgpConvexDecomposition.h>


// hkgpConvexDecomposition ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpConvexDecomposition)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(IProgress)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(AttachedData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(GuardGenConfig)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MeshPreProcessorConfig)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Config)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkgpConvexDecomposition, s_libraryName)


// IProgress hkgpConvexDecomposition

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpConvexDecomposition::IProgress)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkgpConvexDecomposition::IProgress)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkgpConvexDecomposition::IProgress, s_libraryName)


// AttachedData hkgpConvexDecomposition

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpConvexDecomposition::AttachedData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkgpConvexDecomposition::AttachedData)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkgpConvexDecomposition::AttachedData, s_libraryName, hkgpConvexHull::IUserObject)


// GuardGenConfig hkgpConvexDecomposition
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkgpConvexDecomposition, GuardGenConfig, s_libraryName)


// MeshPreProcessorConfig hkgpConvexDecomposition
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkgpConvexDecomposition, MeshPreProcessorConfig, s_libraryName)


// Config hkgpConvexDecomposition

HK_TRACKER_DECLARE_CLASS_BEGIN(hkgpConvexDecomposition::Config)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ePrimaryMethod)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(eReduceMethod)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(eHollowParts)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkgpConvexDecomposition::Config)
    HK_TRACKER_MEMBER(hkgpConvexDecomposition::Config, m_sphereGuards, 0, "hkArray<hkVector4f, hkContainerHeapAllocator>") // hkArray< hkVector4f, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkgpConvexDecomposition::Config, m_edgeGuards, 0, "hkArray<hkVector4f, hkContainerHeapAllocator>") // hkArray< hkVector4f, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkgpConvexDecomposition::Config, m_iprogress, 0, "hkgpConvexDecomposition::IProgress*") // const struct hkgpConvexDecomposition::IProgress*
    HK_TRACKER_MEMBER(hkgpConvexDecomposition::Config, m_internal, 0, "void*") // const void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkgpConvexDecomposition::Config, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkgpConvexDecomposition::Config, ePrimaryMethod, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkgpConvexDecomposition::Config, eReduceMethod, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkgpConvexDecomposition::Config, eHollowParts, s_libraryName)

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
