/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#include <Physics2012/Collide/hkpCollide.h>
static const char s_libraryName[] = "hkpCollidehkpCompressedMeshShape";
#include <Common/Base/Memory/Tracker/hkTrackerClassDefinition.h>

void HK_CALL hkpCollidehkpCompressedMeshShapeRegister() {}

#include <Physics2012/Collide/Shape/Deprecated/CompressedMesh/hkpCompressedMeshShape.h>


// hkpCompressedMeshShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCompressedMeshShape)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Chunk)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BigTriangle)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ConvexPiece)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MaterialType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCompressedMeshShape)
    HK_TRACKER_MEMBER(hkpCompressedMeshShape, m_materials, 0, "hkArray<hkUint32, hkContainerHeapAllocator>") // hkArray< hkUint32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpCompressedMeshShape, m_materials16, 0, "hkArray<hkUint16, hkContainerHeapAllocator>") // hkArray< hkUint16, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpCompressedMeshShape, m_materials8, 0, "hkArray<hkUint8, hkContainerHeapAllocator>") // hkArray< hkUint8, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpCompressedMeshShape, m_transforms, 0, "hkArray<hkQsTransformf, hkContainerHeapAllocator>") // hkArray< hkQsTransformf, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpCompressedMeshShape, m_bigVertices, 0, "hkArray<hkVector4f, hkContainerHeapAllocator>") // hkArray< hkVector4f, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpCompressedMeshShape, m_bigTriangles, 0, "hkArray<hkpCompressedMeshShape::BigTriangle, hkContainerHeapAllocator>") // hkArray< struct hkpCompressedMeshShape::BigTriangle, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpCompressedMeshShape, m_chunks, 0, "hkArray<hkpCompressedMeshShape::Chunk, hkContainerHeapAllocator>") // hkArray< class hkpCompressedMeshShape::Chunk, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpCompressedMeshShape, m_convexPieces, 0, "hkArray<hkpCompressedMeshShape::ConvexPiece, hkContainerHeapAllocator>") // hkArray< struct hkpCompressedMeshShape::ConvexPiece, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpCompressedMeshShape, m_meshMaterials, 0, "hkpMeshMaterial*") // class hkpMeshMaterial*
    HK_TRACKER_MEMBER(hkpCompressedMeshShape, m_namedMaterials, 0, "hkArray<hkpNamedMeshMaterial, hkContainerHeapAllocator>") // hkArray< class hkpNamedMeshMaterial, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpCompressedMeshShape, s_libraryName, hkpShapeCollection)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpCompressedMeshShape, MaterialType, s_libraryName)


// Chunk hkpCompressedMeshShape

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCompressedMeshShape::Chunk)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCompressedMeshShape::Chunk)
    HK_TRACKER_MEMBER(hkpCompressedMeshShape::Chunk, m_vertices, 0, "hkArray<hkUint16, hkContainerHeapAllocator>") // hkArray< hkUint16, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpCompressedMeshShape::Chunk, m_indices, 0, "hkArray<hkUint16, hkContainerHeapAllocator>") // hkArray< hkUint16, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpCompressedMeshShape::Chunk, m_stripLengths, 0, "hkArray<hkUint16, hkContainerHeapAllocator>") // hkArray< hkUint16, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpCompressedMeshShape::Chunk, m_weldingInfo, 0, "hkArray<hkUint16, hkContainerHeapAllocator>") // hkArray< hkUint16, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpCompressedMeshShape::Chunk, s_libraryName)


// BigTriangle hkpCompressedMeshShape
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpCompressedMeshShape, BigTriangle, s_libraryName)


// ConvexPiece hkpCompressedMeshShape

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCompressedMeshShape::ConvexPiece)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCompressedMeshShape::ConvexPiece)
    HK_TRACKER_MEMBER(hkpCompressedMeshShape::ConvexPiece, m_vertices, 0, "hkArray<hkUint16, hkContainerHeapAllocator>") // hkArray< hkUint16, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpCompressedMeshShape::ConvexPiece, s_libraryName)

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
