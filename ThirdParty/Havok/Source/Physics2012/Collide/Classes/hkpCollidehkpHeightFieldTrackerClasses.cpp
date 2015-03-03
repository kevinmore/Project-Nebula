/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#include <Physics2012/Collide/hkpCollide.h>
static const char s_libraryName[] = "hkpCollidehkpHeightField";
#include <Common/Base/Memory/Tracker/hkTrackerClassDefinition.h>

void HK_CALL hkpCollidehkpHeightFieldRegister() {}

#include <Physics2012/Collide/Shape/HeightField/CompressedSampledHeightField/hkpCompressedSampledHeightFieldShape.h>


// hkpCompressedSampledHeightFieldShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCompressedSampledHeightFieldShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCompressedSampledHeightFieldShape)
    HK_TRACKER_MEMBER(hkpCompressedSampledHeightFieldShape, m_storage, 0, "hkArray<hkUint16, hkContainerHeapAllocator>") // hkArray< hkUint16, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpCompressedSampledHeightFieldShape, s_libraryName, hkpSampledHeightFieldShape)

#include <Physics2012/Collide/Shape/HeightField/Plane/hkpPlaneShape.h>


// hkpPlaneShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPlaneShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPlaneShape)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpPlaneShape, s_libraryName, hkpHeightFieldShape)

#include <Physics2012/Collide/Shape/HeightField/SampledHeightField/hkpSampledHeightFieldShape.h>


// hkpSampledHeightFieldShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSampledHeightFieldShape)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(AABBStackElement)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CoarseMinMaxLevel)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(HeightFieldType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSampledHeightFieldShape)
    HK_TRACKER_MEMBER(hkpSampledHeightFieldShape, m_coarseTreeData, 0, "hkArray<hkpSampledHeightFieldShape::CoarseMinMaxLevel, hkContainerHeapAllocator>") // hkArray< struct hkpSampledHeightFieldShape::CoarseMinMaxLevel, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpSampledHeightFieldShape, s_libraryName, hkpHeightFieldShape)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpSampledHeightFieldShape, HeightFieldType, s_libraryName)


// AABBStackElement hkpSampledHeightFieldShape
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpSampledHeightFieldShape, AABBStackElement, s_libraryName)


// CoarseMinMaxLevel hkpSampledHeightFieldShape

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSampledHeightFieldShape::CoarseMinMaxLevel)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSampledHeightFieldShape::CoarseMinMaxLevel)
    HK_TRACKER_MEMBER(hkpSampledHeightFieldShape::CoarseMinMaxLevel, m_minMaxData, 0, "hkArray<hkVector4f, hkContainerHeapAllocator>") // hkArray< hkVector4f, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpSampledHeightFieldShape::CoarseMinMaxLevel, s_libraryName)

#include <Physics2012/Collide/Shape/HeightField/StorageSampledHeightField/hkpStorageSampledHeightFieldShape.h>


// hkpStorageSampledHeightFieldShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpStorageSampledHeightFieldShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpStorageSampledHeightFieldShape)
    HK_TRACKER_MEMBER(hkpStorageSampledHeightFieldShape, m_storage, 0, "hkArray<float, hkContainerHeapAllocator>") // hkArray< float, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpStorageSampledHeightFieldShape, s_libraryName, hkpSampledHeightFieldShape)

#include <Physics2012/Collide/Shape/HeightField/TriSampledHeightField/hkpTriSampledHeightFieldBvTreeShape.h>


// hkpTriSampledHeightFieldBvTreeShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpTriSampledHeightFieldBvTreeShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpTriSampledHeightFieldBvTreeShape)
    HK_TRACKER_MEMBER(hkpTriSampledHeightFieldBvTreeShape, m_childContainer, 0, "hkpSingleShapeContainer") // class hkpSingleShapeContainer
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpTriSampledHeightFieldBvTreeShape, s_libraryName, hkpBvTreeShape)

#include <Physics2012/Collide/Shape/HeightField/TriSampledHeightField/hkpTriSampledHeightFieldCollection.h>


// hkpTriSampledHeightFieldCollection ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpTriSampledHeightFieldCollection)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpTriSampledHeightFieldCollection)
    HK_TRACKER_MEMBER(hkpTriSampledHeightFieldCollection, m_heightfield, 0, "hkpSampledHeightFieldShape*") // const class hkpSampledHeightFieldShape*
    HK_TRACKER_MEMBER(hkpTriSampledHeightFieldCollection, m_weldingInfo, 0, "hkArray<hkUint16, hkContainerHeapAllocator>") // hkArray< hkUint16, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpTriSampledHeightFieldCollection, s_libraryName, hkpShapeCollection)

#include <Physics2012/Collide/Shape/HeightField/hkpHeightFieldShape.h>


// hkpHeightFieldShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpHeightFieldShape)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CollideSpheresInput)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(hkpSphereCastInput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpHeightFieldShape)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpHeightFieldShape, s_libraryName, hkpShape)


// CollideSpheresInput hkpHeightFieldShape
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpHeightFieldShape, CollideSpheresInput, s_libraryName)


// hkpSphereCastInput hkpHeightFieldShape
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpHeightFieldShape, hkpSphereCastInput, s_libraryName)

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
