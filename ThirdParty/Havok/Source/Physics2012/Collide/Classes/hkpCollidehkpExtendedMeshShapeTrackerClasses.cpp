/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#include <Physics2012/Collide/hkpCollide.h>
static const char s_libraryName[] = "hkpCollidehkpExtendedMeshShape";
#include <Common/Base/Memory/Tracker/hkTrackerClassDefinition.h>

void HK_CALL hkpCollidehkpExtendedMeshShapeRegister() {}

#include <Physics2012/Collide/Shape/Compound/Collection/ExtendedMeshShape/hkpExtendedMeshShape.h>


// hkpExtendedMeshShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpExtendedMeshShape)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Subpart)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(TrianglesSubpart)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ShapesSubpart)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(IndexStridingType)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MaterialIndexStridingType)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SubpartType)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SubpartTypesAndFlags)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpExtendedMeshShape)
    HK_TRACKER_MEMBER(hkpExtendedMeshShape, m_embeddedTrianglesSubpart, 0, "hkpExtendedMeshShape::TrianglesSubpart") // struct hkpExtendedMeshShape::TrianglesSubpart
    HK_TRACKER_MEMBER(hkpExtendedMeshShape, m_materialClass, 0, "hkClass*") // const hkClass*
    HK_TRACKER_MEMBER(hkpExtendedMeshShape, m_trianglesSubparts, 0, "hkArray<hkpExtendedMeshShape::TrianglesSubpart, hkContainerHeapAllocator>") // hkArray< struct hkpExtendedMeshShape::TrianglesSubpart, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpExtendedMeshShape, m_shapesSubparts, 0, "hkArray<hkpExtendedMeshShape::ShapesSubpart, hkContainerHeapAllocator>") // hkArray< struct hkpExtendedMeshShape::ShapesSubpart, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpExtendedMeshShape, m_weldingInfo, 0, "hkArray<hkUint16, hkContainerHeapAllocator>") // hkArray< hkUint16, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpExtendedMeshShape, s_libraryName, hkpShapeCollection)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpExtendedMeshShape, IndexStridingType, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpExtendedMeshShape, MaterialIndexStridingType, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpExtendedMeshShape, SubpartType, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpExtendedMeshShape, SubpartTypesAndFlags, s_libraryName)


// Subpart hkpExtendedMeshShape

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpExtendedMeshShape::Subpart)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpExtendedMeshShape::Subpart)
    HK_TRACKER_MEMBER(hkpExtendedMeshShape::Subpart, m_materialIndexBase, 0, "void*") // const void*
    HK_TRACKER_MEMBER(hkpExtendedMeshShape::Subpart, m_materialBase, 0, "hkpMeshMaterial*") // const class hkpMeshMaterial*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpExtendedMeshShape::Subpart, s_libraryName)


// TrianglesSubpart hkpExtendedMeshShape

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpExtendedMeshShape::TrianglesSubpart)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpExtendedMeshShape::TrianglesSubpart)
    HK_TRACKER_MEMBER(hkpExtendedMeshShape::TrianglesSubpart, m_vertexBase, 0, "float*") // const float*
    HK_TRACKER_MEMBER(hkpExtendedMeshShape::TrianglesSubpart, m_indexBase, 0, "void*") // const void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpExtendedMeshShape::TrianglesSubpart, s_libraryName, hkpExtendedMeshShape::Subpart)


// ShapesSubpart hkpExtendedMeshShape

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpExtendedMeshShape::ShapesSubpart)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpExtendedMeshShape::ShapesSubpart)
    HK_TRACKER_MEMBER(hkpExtendedMeshShape::ShapesSubpart, m_childShapes, 0, "hkArray<hkpConvexShape *, hkContainerHeapAllocator>") // hkArray< class hkpConvexShape *, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpExtendedMeshShape::ShapesSubpart, s_libraryName, hkpExtendedMeshShape::Subpart)

#include <Physics2012/Collide/Shape/Compound/Collection/StorageExtendedMesh/hkpStorageExtendedMeshShape.h>


// hkpStorageExtendedMeshShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpStorageExtendedMeshShape)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Material)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MeshSubpartStorage)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ShapeSubpartStorage)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpStorageExtendedMeshShape)
    HK_TRACKER_MEMBER(hkpStorageExtendedMeshShape, m_meshstorage, 0, "hkArray<hkpStorageExtendedMeshShape::MeshSubpartStorage*, hkContainerHeapAllocator>") // hkArray< struct hkpStorageExtendedMeshShape::MeshSubpartStorage*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpStorageExtendedMeshShape, m_shapestorage, 0, "hkArray<hkpStorageExtendedMeshShape::ShapeSubpartStorage*, hkContainerHeapAllocator>") // hkArray< struct hkpStorageExtendedMeshShape::ShapeSubpartStorage*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpStorageExtendedMeshShape, s_libraryName, hkpExtendedMeshShape)


// Material hkpStorageExtendedMeshShape
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpStorageExtendedMeshShape, Material, s_libraryName)


// MeshSubpartStorage hkpStorageExtendedMeshShape

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpStorageExtendedMeshShape::MeshSubpartStorage)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpStorageExtendedMeshShape::MeshSubpartStorage)
    HK_TRACKER_MEMBER(hkpStorageExtendedMeshShape::MeshSubpartStorage, m_vertices, 0, "hkArray<hkVector4f, hkContainerHeapAllocator>") // hkArray< hkVector4f, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpStorageExtendedMeshShape::MeshSubpartStorage, m_indices8, 0, "hkArray<hkUint8, hkContainerHeapAllocator>") // hkArray< hkUint8, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpStorageExtendedMeshShape::MeshSubpartStorage, m_indices16, 0, "hkArray<hkUint16, hkContainerHeapAllocator>") // hkArray< hkUint16, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpStorageExtendedMeshShape::MeshSubpartStorage, m_indices32, 0, "hkArray<hkUint32, hkContainerHeapAllocator>") // hkArray< hkUint32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpStorageExtendedMeshShape::MeshSubpartStorage, m_materialIndices, 0, "hkArray<hkUint8, hkContainerHeapAllocator>") // hkArray< hkUint8, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpStorageExtendedMeshShape::MeshSubpartStorage, m_materials, 0, "hkArray<hkpStorageExtendedMeshShape::Material, hkContainerHeapAllocator>") // hkArray< struct hkpStorageExtendedMeshShape::Material, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpStorageExtendedMeshShape::MeshSubpartStorage, m_namedMaterials, 0, "hkArray<hkpNamedMeshMaterial, hkContainerHeapAllocator>") // hkArray< class hkpNamedMeshMaterial, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpStorageExtendedMeshShape::MeshSubpartStorage, m_materialIndices16, 0, "hkArray<hkUint16, hkContainerHeapAllocator>") // hkArray< hkUint16, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpStorageExtendedMeshShape::MeshSubpartStorage, s_libraryName, hkReferencedObject)


// ShapeSubpartStorage hkpStorageExtendedMeshShape

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpStorageExtendedMeshShape::ShapeSubpartStorage)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpStorageExtendedMeshShape::ShapeSubpartStorage)
    HK_TRACKER_MEMBER(hkpStorageExtendedMeshShape::ShapeSubpartStorage, m_materialIndices, 0, "hkArray<hkUint8, hkContainerHeapAllocator>") // hkArray< hkUint8, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpStorageExtendedMeshShape::ShapeSubpartStorage, m_materials, 0, "hkArray<hkpStorageExtendedMeshShape::Material, hkContainerHeapAllocator>") // hkArray< struct hkpStorageExtendedMeshShape::Material, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpStorageExtendedMeshShape::ShapeSubpartStorage, m_materialIndices16, 0, "hkArray<hkUint16, hkContainerHeapAllocator>") // hkArray< hkUint16, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpStorageExtendedMeshShape::ShapeSubpartStorage, s_libraryName, hkReferencedObject)

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
