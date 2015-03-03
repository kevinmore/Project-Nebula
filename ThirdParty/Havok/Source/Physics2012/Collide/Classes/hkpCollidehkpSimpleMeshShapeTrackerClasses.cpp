/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#include <Physics2012/Collide/hkpCollide.h>
static const char s_libraryName[] = "hkpCollidehkpSimpleMeshShape";
#include <Common/Base/Memory/Tracker/hkTrackerClassDefinition.h>

void HK_CALL hkpCollidehkpSimpleMeshShapeRegister() {}

#include <Physics2012/Collide/Shape/Compound/Collection/SimpleMesh/hkpSimpleMeshShape.h>


// hkpSimpleMeshShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSimpleMeshShape)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Triangle)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSimpleMeshShape)
    HK_TRACKER_MEMBER(hkpSimpleMeshShape, m_vertices, 0, "hkArray<hkVector4f, hkContainerHeapAllocator>") // hkArray< hkVector4f, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpSimpleMeshShape, m_triangles, 0, "hkArray<hkpSimpleMeshShape::Triangle, hkContainerHeapAllocator>") // hkArray< struct hkpSimpleMeshShape::Triangle, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpSimpleMeshShape, m_materialIndices, 0, "hkArray<hkUint8, hkContainerHeapAllocator>") // hkArray< hkUint8, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpSimpleMeshShape, s_libraryName, hkpShapeCollection)


// Triangle hkpSimpleMeshShape
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpSimpleMeshShape, Triangle, s_libraryName)

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
