/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#include <Physics2012/Collide/hkpCollide.h>
static const char s_libraryName[] = "hkpCollidehkpMeshShape";
#include <Common/Base/Memory/Tracker/hkTrackerClassDefinition.h>

void HK_CALL hkpCollidehkpMeshShapeRegister() {}

#include <Physics2012/Collide/Shape/Deprecated/FastMesh/hkpFastMeshShape.h>


// hkpFastMeshShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpFastMeshShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpFastMeshShape)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpFastMeshShape, s_libraryName, hkpMeshShape)

#include <Physics2012/Collide/Shape/Deprecated/Mesh/hkpMeshShape.h>


// hkpMeshShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMeshShape)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Subpart)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MeshShapeIndexStridingType)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MeshShapeMaterialIndexStridingType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMeshShape)
    HK_TRACKER_MEMBER(hkpMeshShape, m_subparts, 0, "hkArray<hkpMeshShape::Subpart, hkContainerHeapAllocator>") // hkArray< struct hkpMeshShape::Subpart, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpMeshShape, m_weldingInfo, 0, "hkArray<hkUint16, hkContainerHeapAllocator>") // hkArray< hkUint16, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpMeshShape, s_libraryName, hkpShapeCollection)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpMeshShape, MeshShapeIndexStridingType, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpMeshShape, MeshShapeMaterialIndexStridingType, s_libraryName)


// Subpart hkpMeshShape

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMeshShape::Subpart)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMeshShape::Subpart)
    HK_TRACKER_MEMBER(hkpMeshShape::Subpart, m_vertexBase, 0, "float*") // const float*
    HK_TRACKER_MEMBER(hkpMeshShape::Subpart, m_indexBase, 0, "void*") // const void*
    HK_TRACKER_MEMBER(hkpMeshShape::Subpart, m_materialIndexBase, 0, "void*") // const void*
    HK_TRACKER_MEMBER(hkpMeshShape::Subpart, m_materialBase, 0, "hkpMeshMaterial*") // const class hkpMeshMaterial*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpMeshShape::Subpart, s_libraryName)

#include <Physics2012/Collide/Shape/Deprecated/StorageMesh/hkpStorageMeshShape.h>


// hkpStorageMeshShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpStorageMeshShape)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SubpartStorage)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpStorageMeshShape)
    HK_TRACKER_MEMBER(hkpStorageMeshShape, m_storage, 0, "hkArray<hkpStorageMeshShape::SubpartStorage*, hkContainerHeapAllocator>") // hkArray< struct hkpStorageMeshShape::SubpartStorage*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpStorageMeshShape, s_libraryName, hkpMeshShape)


// SubpartStorage hkpStorageMeshShape

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpStorageMeshShape::SubpartStorage)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpStorageMeshShape::SubpartStorage)
    HK_TRACKER_MEMBER(hkpStorageMeshShape::SubpartStorage, m_vertices, 0, "hkArray<float, hkContainerHeapAllocator>") // hkArray< float, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpStorageMeshShape::SubpartStorage, m_indices16, 0, "hkArray<hkUint16, hkContainerHeapAllocator>") // hkArray< hkUint16, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpStorageMeshShape::SubpartStorage, m_indices32, 0, "hkArray<hkUint32, hkContainerHeapAllocator>") // hkArray< hkUint32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpStorageMeshShape::SubpartStorage, m_materialIndices, 0, "hkArray<hkUint8, hkContainerHeapAllocator>") // hkArray< hkUint8, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpStorageMeshShape::SubpartStorage, m_materials, 0, "hkArray<hkUint32, hkContainerHeapAllocator>") // hkArray< hkUint32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpStorageMeshShape::SubpartStorage, m_materialIndices16, 0, "hkArray<hkUint16, hkContainerHeapAllocator>") // hkArray< hkUint16, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpStorageMeshShape::SubpartStorage, s_libraryName, hkReferencedObject)

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
