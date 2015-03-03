/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#include <Geometry/Internal/hkcdInternal.h>
static const char s_libraryName[] = "hkcdInternal";
#include <Common/Base/Memory/Tracker/hkTrackerClassDefinition.h>

void HK_CALL hkcdInternalRegister() {}

#include <Geometry/Internal/Algorithms/Particle/hkcdParticleTriangleUtil.h>


// hkcdParticleTriangleUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkcdParticleTriangleUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(TriangleCollisionInfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ParticleInfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(TriangleCache)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkcdParticleTriangleUtil, s_libraryName)


// TriangleCollisionInfo hkcdParticleTriangleUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkcdParticleTriangleUtil, TriangleCollisionInfo, s_libraryName)


// ParticleInfo hkcdParticleTriangleUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkcdParticleTriangleUtil, ParticleInfo, s_libraryName)


// TriangleCache hkcdParticleTriangleUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkcdParticleTriangleUtil, TriangleCache, s_libraryName)

#include <Geometry/Internal/Algorithms/TreeQueries/hkcdAabbTreeQueries.h>


// hkcdAabbTreeQueries ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkcdAabbTreeQueries)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(RaycastCollector)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(AabbCollector)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ClosestPointCollector)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(AllPairsCollector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkcdAabbTreeQueries, s_libraryName)


// RaycastCollector hkcdAabbTreeQueries

HK_TRACKER_DECLARE_CLASS_BEGIN(hkcdAabbTreeQueries::RaycastCollector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkcdAabbTreeQueries::RaycastCollector)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkcdAabbTreeQueries::RaycastCollector, s_libraryName)


// AabbCollector hkcdAabbTreeQueries

HK_TRACKER_DECLARE_CLASS_BEGIN(hkcdAabbTreeQueries::AabbCollector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkcdAabbTreeQueries::AabbCollector)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkcdAabbTreeQueries::AabbCollector, s_libraryName)


// ClosestPointCollector hkcdAabbTreeQueries

HK_TRACKER_DECLARE_CLASS_BEGIN(hkcdAabbTreeQueries::ClosestPointCollector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkcdAabbTreeQueries::ClosestPointCollector)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkcdAabbTreeQueries::ClosestPointCollector, s_libraryName)


// AllPairsCollector hkcdAabbTreeQueries

HK_TRACKER_DECLARE_CLASS_BEGIN(hkcdAabbTreeQueries::AllPairsCollector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkcdAabbTreeQueries::AllPairsCollector)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkcdAabbTreeQueries::AllPairsCollector, s_libraryName)

#include <Geometry/Internal/DataStructures/DynamicTree/hkcdDynamicAabbTree.h>


// hkcdDynamicAabbTree ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkcdDynamicAabbTree)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkcdDynamicAabbTree)
    HK_TRACKER_MEMBER(hkcdDynamicAabbTree, m_treePtr, 0, "void*") // void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkcdDynamicAabbTree, s_libraryName, hkReferencedObject)

#include <Geometry/Internal/DataStructures/StaticTree/hkcdStaticAabbTree.h>


// hkcdStaticAabbTree ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkcdStaticAabbTree)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkcdStaticAabbTree)
    HK_TRACKER_MEMBER(hkcdStaticAabbTree, m_treePtr, 0, "void*") // void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkcdStaticAabbTree, s_libraryName, hkReferencedObject)

#include <Geometry/Internal/Types/hkcdVertex.h>


// hkcdVertex ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkcdVertex, s_libraryName)

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
