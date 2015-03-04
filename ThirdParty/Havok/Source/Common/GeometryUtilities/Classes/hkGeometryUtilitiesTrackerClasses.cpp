/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#include <Common/GeometryUtilities/hkGeometryUtilities.h>
static const char s_libraryName[] = "hkGeometryUtilities";
#include <Common/Base/Memory/Tracker/hkTrackerClassDefinition.h>

void HK_CALL hkGeometryUtilitiesRegister() {}

#include <Common/GeometryUtilities/Inertia/hkCompressedInertiaTensor.h>


// hkDiagonalizedMassProperties ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkDiagonalizedMassProperties, s_libraryName)


// hkCompressedMassProperties ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkCompressedMassProperties, s_libraryName)

#include <Common/GeometryUtilities/Inertia/hkInertiaTensorComputer.h>


// hkMassProperties ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkMassProperties, s_libraryName)


// hkMassElement ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkMassElement, s_libraryName)


// hkInertiaTensorComputer ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkInertiaTensorComputer, s_libraryName)

#include <Common/GeometryUtilities/Matching/hkGeometryMatchingUtils.h>


// hkGeometryMatchingUtils ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkGeometryMatchingUtils)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Geometry)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(TriangleMap)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(FullMap)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkGeometryMatchingUtils, s_libraryName)


// Geometry hkGeometryMatchingUtils

HK_TRACKER_DECLARE_CLASS_BEGIN(hkGeometryMatchingUtils::Geometry)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkGeometryMatchingUtils::Geometry)
    HK_TRACKER_MEMBER(hkGeometryMatchingUtils::Geometry, m_vertices, 0, "hkVector4*") // const hkVector4*
    HK_TRACKER_MEMBER(hkGeometryMatchingUtils::Geometry, m_triangleIndices, 0, "hkInt32*") // const hkInt32*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkGeometryMatchingUtils::Geometry, s_libraryName)


// TriangleMap hkGeometryMatchingUtils

HK_TRACKER_DECLARE_CLASS_BEGIN(hkGeometryMatchingUtils::TriangleMap)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Hit)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkGeometryMatchingUtils::TriangleMap)
    HK_TRACKER_MEMBER(hkGeometryMatchingUtils::TriangleMap, m_startIndexPerGeometry, 0, "hkInplaceArray<hkUint32, 16, hkContainerHeapAllocator>") // class hkInplaceArray< hkUint32, 16, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkGeometryMatchingUtils::TriangleMap, m_foundReferenceTriangle, 0, "hkInplaceArray<hkGeometryMatchingUtils::TriangleMap::Hit, 128, hkContainerHeapAllocator>") // class hkInplaceArray< struct hkGeometryMatchingUtils::TriangleMap::Hit, 128, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkGeometryMatchingUtils::TriangleMap, s_libraryName)


// Hit hkGeometryMatchingUtils::TriangleMap
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkGeometryMatchingUtils::TriangleMap, Hit, s_libraryName)


// FullMap hkGeometryMatchingUtils

HK_TRACKER_DECLARE_CLASS_BEGIN(hkGeometryMatchingUtils::FullMap)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(VertexHit)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(VertexTriangleEntry)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkGeometryMatchingUtils::FullMap)
    HK_TRACKER_MEMBER(hkGeometryMatchingUtils::FullMap, m_triangleMap, 0, "hkGeometryMatchingUtils::TriangleMap") // struct hkGeometryMatchingUtils::TriangleMap
    HK_TRACKER_MEMBER(hkGeometryMatchingUtils::FullMap, m_startEntryPerGeometry, 0, "hkInplaceArray<hkUint32, 16, hkContainerHeapAllocator>") // class hkInplaceArray< hkUint32, 16, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkGeometryMatchingUtils::FullMap, m_searchTrianglePerSearchVertex, 0, "hkArray<hkGeometryMatchingUtils::FullMap::VertexTriangleEntry, hkContainerHeapAllocator>") // hkArray< struct hkGeometryMatchingUtils::FullMap::VertexTriangleEntry, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkGeometryMatchingUtils::FullMap, s_libraryName)


// VertexHit hkGeometryMatchingUtils::FullMap
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkGeometryMatchingUtils::FullMap, VertexHit, s_libraryName)


// VertexTriangleEntry hkGeometryMatchingUtils::FullMap
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkGeometryMatchingUtils::FullMap, VertexTriangleEntry, s_libraryName)

#include <Common/GeometryUtilities/Mesh/Converters/MeshToSceneData/hkMeshToSceneDataConverter.h>


// hkMeshToSceneDataConverter ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkMeshToSceneDataConverter, s_libraryName)

#include <Common/GeometryUtilities/Mesh/Converters/MeshTohkGeometry/hkMeshTohkGeometryConverter.h>


// hkMeshTohkGeometryConverter ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkMeshTohkGeometryConverter, s_libraryName)

#include <Common/GeometryUtilities/Mesh/Converters/SceneDataToMesh/hkSceneDataToMeshConverter.h>


// hkSceneDataToMeshConverter ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkSceneDataToMeshConverter, s_libraryName)

#include <Common/GeometryUtilities/Mesh/Default/hkDefaultCompoundMeshShape.h>


// hkDefaultCompoundMeshShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDefaultCompoundMeshShape)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MeshSection)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDefaultCompoundMeshShape)
    HK_TRACKER_MEMBER(hkDefaultCompoundMeshShape, m_shapes, 0, "hkArray<hkMeshShape*, hkContainerHeapAllocator>") // hkArray< const class hkMeshShape*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkDefaultCompoundMeshShape, m_defaultChildTransforms, 0, "hkArray<hkMatrix4f, hkContainerHeapAllocator>") // hkArray< hkMatrix4f, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkDefaultCompoundMeshShape, m_sections, 0, "hkArray<hkDefaultCompoundMeshShape::MeshSection, hkContainerHeapAllocator>") // hkArray< struct hkDefaultCompoundMeshShape::MeshSection, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkDefaultCompoundMeshShape, s_libraryName, hkMeshShape)


// MeshSection hkDefaultCompoundMeshShape
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkDefaultCompoundMeshShape, MeshSection, s_libraryName)


// hkDefaultCompoundMeshBody ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDefaultCompoundMeshBody)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDefaultCompoundMeshBody)
    HK_TRACKER_MEMBER(hkDefaultCompoundMeshBody, m_bodies, 0, "hkArray<hkMeshBody*, hkContainerHeapAllocator>") // hkArray< class hkMeshBody*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkDefaultCompoundMeshBody, m_shape, 0, "hkDefaultCompoundMeshShape*") // const class hkDefaultCompoundMeshShape*
    HK_TRACKER_MEMBER(hkDefaultCompoundMeshBody, m_transformSet, 0, "hkIndexedTransformSet *") // class hkIndexedTransformSet *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkDefaultCompoundMeshBody, s_libraryName, hkMeshBody)

#include <Common/GeometryUtilities/Mesh/Default/hkDefaultMeshMaterialRegistry.h>


// hkDefaultMeshMaterialRegistry ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDefaultMeshMaterialRegistry)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Entry)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDefaultMeshMaterialRegistry)
    HK_TRACKER_MEMBER(hkDefaultMeshMaterialRegistry, m_entries, 0, "hkArray<hkDefaultMeshMaterialRegistry::Entry, hkContainerHeapAllocator>") // hkArray< struct hkDefaultMeshMaterialRegistry::Entry, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkDefaultMeshMaterialRegistry, s_libraryName, hkMeshMaterialRegistry)


// Entry hkDefaultMeshMaterialRegistry

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDefaultMeshMaterialRegistry::Entry)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDefaultMeshMaterialRegistry::Entry)
    HK_TRACKER_MEMBER(hkDefaultMeshMaterialRegistry::Entry, m_name, 0, "hkStringPtr") // hkStringPtr
    HK_TRACKER_MEMBER(hkDefaultMeshMaterialRegistry::Entry, m_material, 0, "hkMeshMaterial *") // class hkMeshMaterial *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkDefaultMeshMaterialRegistry::Entry, s_libraryName)

#include <Common/GeometryUtilities/Mesh/Deform/hkSimdSkinningDeformer.h>


// hkSimdSkinningDeformer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkSimdSkinningDeformer)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Binding)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkSimdSkinningDeformer)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkSimdSkinningDeformer, s_libraryName, hkReferencedObject)


// Binding hkSimdSkinningDeformer

HK_TRACKER_DECLARE_CLASS_BEGIN(hkSimdSkinningDeformer::Binding)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkSimdSkinningDeformer::Binding)
    HK_TRACKER_MEMBER(hkSimdSkinningDeformer::Binding, m_input, 0, "hkMeshVertexBuffer::LockedVertices") // struct hkMeshVertexBuffer::LockedVertices
    HK_TRACKER_MEMBER(hkSimdSkinningDeformer::Binding, m_output, 0, "hkMeshVertexBuffer::LockedVertices") // struct hkMeshVertexBuffer::LockedVertices
    HK_TRACKER_MEMBER(hkSimdSkinningDeformer::Binding, m_weights, 0, "hkUint8*") // const hkUint8*
    HK_TRACKER_MEMBER(hkSimdSkinningDeformer::Binding, m_transformIndices8, 0, "hkUint8*") // const hkUint8*
    HK_TRACKER_MEMBER(hkSimdSkinningDeformer::Binding, m_transformIndices16, 0, "hkUint16*") // const hkUint16*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkSimdSkinningDeformer::Binding, s_libraryName)

#include <Common/GeometryUtilities/Mesh/Deform/hkSkinOperator.h>


// hkSkinOperator ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkSkinOperator)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Buffer)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BoneInfluence)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Component)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SkinStream)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Parameters)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ExecutionFlags)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkSkinOperator, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkSkinOperator, ExecutionFlags, s_libraryName)


// Buffer hkSkinOperator

HK_TRACKER_DECLARE_CLASS_BEGIN(hkSkinOperator::Buffer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkSkinOperator::Buffer)
    HK_TRACKER_MEMBER(hkSkinOperator::Buffer, m_start, 0, "void*") // void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkSkinOperator::Buffer, s_libraryName)


// BoneInfluence hkSkinOperator
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkSkinOperator, BoneInfluence, s_libraryName)


// Component hkSkinOperator
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkSkinOperator, Component, s_libraryName)


// SkinStream hkSkinOperator

HK_TRACKER_DECLARE_CLASS_BEGIN(hkSkinOperator::SkinStream)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkSkinOperator::SkinStream)
    HK_TRACKER_MEMBER(hkSkinOperator::SkinStream, m_buffers, 0, "hkSkinOperator::Buffer [4]") // struct hkSkinOperator::Buffer [4]
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkSkinOperator::SkinStream, s_libraryName)


// Parameters hkSkinOperator

HK_TRACKER_DECLARE_CLASS_BEGIN(hkSkinOperator::Parameters)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkSkinOperator::Parameters)
    HK_TRACKER_MEMBER(hkSkinOperator::Parameters, m_compositeMatrices, 0, "hkMatrix4*") // const hkMatrix4*
    HK_TRACKER_MEMBER(hkSkinOperator::Parameters, m_boneInfluences, 0, "hkSkinOperator::BoneInfluence*") // const struct hkSkinOperator::BoneInfluence*
    HK_TRACKER_MEMBER(hkSkinOperator::Parameters, m_startInfluencePerVertex, 0, "hkUint16*") // const hkUint16*
    HK_TRACKER_MEMBER(hkSkinOperator::Parameters, m_input, 0, "hkSkinOperator::SkinStream") // struct hkSkinOperator::SkinStream
    HK_TRACKER_MEMBER(hkSkinOperator::Parameters, m_output, 0, "hkSkinOperator::SkinStream") // struct hkSkinOperator::SkinStream
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkSkinOperator::Parameters, s_libraryName)

#include <Common/GeometryUtilities/Mesh/IndexedTransformSet/hkIndexedTransformSet.h>


// hkMeshBoneIndexMapping ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMeshBoneIndexMapping)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMeshBoneIndexMapping)
    HK_TRACKER_MEMBER(hkMeshBoneIndexMapping, m_mapping, 0, "hkArray<hkInt16, hkContainerHeapAllocator>") // hkArray< hkInt16, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkMeshBoneIndexMapping, s_libraryName)


// hkIndexedTransformSetCinfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkIndexedTransformSetCinfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkIndexedTransformSetCinfo)
    HK_TRACKER_MEMBER(hkIndexedTransformSetCinfo, m_inverseMatrices, 0, "hkMatrix4*") // const hkMatrix4*
    HK_TRACKER_MEMBER(hkIndexedTransformSetCinfo, m_matrices, 0, "hkMatrix4*") // const hkMatrix4*
    HK_TRACKER_MEMBER(hkIndexedTransformSetCinfo, m_matricesOrder, 0, "hkInt16*") // const hkInt16*
    HK_TRACKER_MEMBER(hkIndexedTransformSetCinfo, m_matricesNames, 0, "hkStringPtr*") // const hkStringPtr*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkIndexedTransformSetCinfo, s_libraryName)


// hkIndexedTransformSet ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkIndexedTransformSet)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkIndexedTransformSet)
    HK_TRACKER_MEMBER(hkIndexedTransformSet, m_matrices, 0, "hkArray<hkMatrix4f, hkContainerHeapAllocator>") // hkArray< hkMatrix4f, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkIndexedTransformSet, m_inverseMatrices, 0, "hkArray<hkMatrix4f, hkContainerHeapAllocator>") // hkArray< hkMatrix4f, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkIndexedTransformSet, m_matricesOrder, 0, "hkArray<hkInt16, hkContainerHeapAllocator>") // hkArray< hkInt16, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkIndexedTransformSet, m_matricesNames, 0, "hkArray<hkStringPtr, hkContainerHeapAllocator>") // hkArray< hkStringPtr, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkIndexedTransformSet, m_indexMappings, 0, "hkArray<hkMeshBoneIndexMapping, hkContainerHeapAllocator>") // hkArray< struct hkMeshBoneIndexMapping, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkIndexedTransformSet, s_libraryName, hkReferencedObject)

#include <Common/GeometryUtilities/Mesh/Memory/hkMemoryMeshBody.h>


// hkMemoryMeshBody ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMemoryMeshBody)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMemoryMeshBody)
    HK_TRACKER_MEMBER(hkMemoryMeshBody, m_transformSet, 0, "hkIndexedTransformSet *") // class hkIndexedTransformSet *
    HK_TRACKER_MEMBER(hkMemoryMeshBody, m_shape, 0, "hkMeshShape *") // const class hkMeshShape *
    HK_TRACKER_MEMBER(hkMemoryMeshBody, m_vertexBuffers, 0, "hkArray<hkMeshVertexBuffer*, hkContainerHeapAllocator>") // hkArray< class hkMeshVertexBuffer*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkMemoryMeshBody, m_name, 0, "hkStringPtr") // hkStringPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkMemoryMeshBody, s_libraryName, hkMeshBody)

#include <Common/GeometryUtilities/Mesh/Memory/hkMemoryMeshMaterial.h>


// hkMemoryMeshMaterial ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMemoryMeshMaterial)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMemoryMeshMaterial)
    HK_TRACKER_MEMBER(hkMemoryMeshMaterial, m_materialName, 0, "hkStringPtr") // hkStringPtr
    HK_TRACKER_MEMBER(hkMemoryMeshMaterial, m_textures, 0, "hkArray<hkMeshTexture *, hkContainerHeapAllocator>") // hkArray< class hkMeshTexture *, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkMemoryMeshMaterial, s_libraryName, hkMeshMaterial)

#include <Common/GeometryUtilities/Mesh/Memory/hkMemoryMeshShape.h>


// hkMemoryMeshShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMemoryMeshShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMemoryMeshShape)
    HK_TRACKER_MEMBER(hkMemoryMeshShape, m_sections, 0, "hkArray<hkMeshSectionCinfo, hkContainerHeapAllocator>") // hkArray< struct hkMeshSectionCinfo, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkMemoryMeshShape, m_indices16, 0, "hkArray<hkUint16, hkContainerHeapAllocator>") // hkArray< hkUint16, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkMemoryMeshShape, m_indices32, 0, "hkArray<hkUint32, hkContainerHeapAllocator>") // hkArray< hkUint32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkMemoryMeshShape, m_name, 0, "hkStringPtr") // hkStringPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkMemoryMeshShape, s_libraryName, hkMeshShape)

#include <Common/GeometryUtilities/Mesh/Memory/hkMemoryMeshSystem.h>


// hkMemoryMeshSystem ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMemoryMeshSystem)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMemoryMeshSystem)
    HK_TRACKER_MEMBER(hkMemoryMeshSystem, m_bodies, 0, "hkPointerMap<hkMeshBody*, hkInt32, hkContainerHeapAllocator>") // class hkPointerMap< class hkMeshBody*, hkInt32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkMemoryMeshSystem, m_materialRegistry, 0, "hkMeshMaterialRegistry *") // class hkMeshMaterialRegistry *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkMemoryMeshSystem, s_libraryName, hkMeshSystem)

#include <Common/GeometryUtilities/Mesh/Memory/hkMemoryMeshTexture.h>


// hkMemoryMeshTexture ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMemoryMeshTexture)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMemoryMeshTexture)
    HK_TRACKER_MEMBER(hkMemoryMeshTexture, m_filename, 0, "hkStringPtr") // hkStringPtr
    HK_TRACKER_MEMBER(hkMemoryMeshTexture, m_data, 0, "hkArray<hkUint8, hkContainerHeapAllocator>") // hkArray< hkUint8, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkMemoryMeshTexture, s_libraryName, hkMeshTexture)

#include <Common/GeometryUtilities/Mesh/Memory/hkMemoryMeshVertexBuffer.h>


// hkMemoryMeshVertexBuffer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMemoryMeshVertexBuffer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMemoryMeshVertexBuffer)
    HK_TRACKER_MEMBER(hkMemoryMeshVertexBuffer, m_memory, 0, "hkArray<hkUint8, hkContainerHeapAllocator>") // hkArray< hkUint8, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkMemoryMeshVertexBuffer, s_libraryName, hkMeshVertexBuffer)

#include <Common/GeometryUtilities/Mesh/MultipleVertexBuffer/hkMultipleVertexBuffer.h>


// hkMultipleVertexBuffer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMultipleVertexBuffer)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(VertexBufferInfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ElementInfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(LockedElement)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMultipleVertexBuffer)
    HK_TRACKER_MEMBER(hkMultipleVertexBuffer, m_lockedElements, 0, "hkArray<hkMultipleVertexBuffer::LockedElement, hkContainerHeapAllocator>") // hkArray< struct hkMultipleVertexBuffer::LockedElement, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkMultipleVertexBuffer, m_lockedBuffer, 0, "hkMemoryMeshVertexBuffer *") // class hkMemoryMeshVertexBuffer *
    HK_TRACKER_MEMBER(hkMultipleVertexBuffer, m_elementInfos, 0, "hkArray<hkMultipleVertexBuffer::ElementInfo, hkContainerHeapAllocator>") // hkArray< struct hkMultipleVertexBuffer::ElementInfo, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkMultipleVertexBuffer, m_vertexBufferInfos, 0, "hkArray<hkMultipleVertexBuffer::VertexBufferInfo, hkContainerHeapAllocator>") // hkArray< struct hkMultipleVertexBuffer::VertexBufferInfo, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkMultipleVertexBuffer, s_libraryName, hkMeshVertexBuffer)


// VertexBufferInfo hkMultipleVertexBuffer

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMultipleVertexBuffer::VertexBufferInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMultipleVertexBuffer::VertexBufferInfo)
    HK_TRACKER_MEMBER(hkMultipleVertexBuffer::VertexBufferInfo, m_vertexBuffer, 0, "hkMeshVertexBuffer *") // class hkMeshVertexBuffer *
    HK_TRACKER_MEMBER(hkMultipleVertexBuffer::VertexBufferInfo, m_lockedVertices, 0, "hkMeshVertexBuffer::LockedVertices*") // struct hkMeshVertexBuffer::LockedVertices*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkMultipleVertexBuffer::VertexBufferInfo, s_libraryName)


// ElementInfo hkMultipleVertexBuffer
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkMultipleVertexBuffer, ElementInfo, s_libraryName)


// LockedElement hkMultipleVertexBuffer
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkMultipleVertexBuffer, LockedElement, s_libraryName)

#include <Common/GeometryUtilities/Mesh/Simplifiers/QemSimplifier/hkMeshSimplifierConverter.h>


// hkMeshSimplifierConverter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMeshSimplifierConverter)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Group)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMeshSimplifierConverter)
    HK_TRACKER_MEMBER(hkMeshSimplifierConverter, m_materials, 0, "hkArray<hkMeshMaterial *, hkContainerHeapAllocator>") // hkArray< class hkMeshMaterial *, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkMeshSimplifierConverter, m_materialWeight, 0, "hkArray<hkSimdFloat32, hkContainerHeapAllocator>") // hkArray< class hkSimdFloat32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkMeshSimplifierConverter, m_groups, 0, "hkArray<hkMeshSimplifierConverter::Group, hkContainerHeapAllocator>") // hkArray< struct hkMeshSimplifierConverter::Group, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkMeshSimplifierConverter, m_weights, 0, "hkArray<hkSimdFloat32, hkContainerHeapAllocator>") // hkArray< class hkSimdFloat32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkMeshSimplifierConverter, m_weightElements, 0, "hkArray<hkUint32, hkContainerHeapAllocator>") // hkArray< hkUint32, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkMeshSimplifierConverter, s_libraryName)


// Group hkMeshSimplifierConverter

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMeshSimplifierConverter::Group)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMeshSimplifierConverter::Group)
    HK_TRACKER_MEMBER(hkMeshSimplifierConverter::Group, m_material, 0, "hkMeshMaterial *") // class hkMeshMaterial *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkMeshSimplifierConverter::Group, s_libraryName)

#include <Common/GeometryUtilities/Mesh/Simplifiers/QemSimplifier/hkQemSimplifier.h>


// hkQemSimplifier ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkQemSimplifier)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(EdgeMap)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ContractionController)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ScaleCalculator)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SizeScaleCalculator)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(AngleScaleCalculator)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(AttributeEntry)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(AttributeFormat)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Group)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(EdgeContraction)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Triangle)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Attribute)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BoundaryEdge)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(AttributeType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkQemSimplifier)
    HK_TRACKER_MEMBER(hkQemSimplifier, m_contractionFreeList, 0, "hkFreeList") // class hkFreeList
    HK_TRACKER_MEMBER(hkQemSimplifier, m_contractions, 0, "hkMinHeap<hkQemSimplifier::EdgeContraction*, hkQemSimplifier::EdgeContraction>") // class hkMinHeap< struct hkQemSimplifier::EdgeContraction*, struct hkQemSimplifier::EdgeContraction >
    HK_TRACKER_MEMBER(hkQemSimplifier, m_edgeContractionMap, 0, "hkPointerMap<hkUint64, hkQemSimplifier::EdgeContraction*, hkContainerHeapAllocator>") // class hkPointerMap< hkUint64, struct hkQemSimplifier::EdgeContraction*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkQemSimplifier, m_groups, 0, "hkArray<hkQemSimplifier::Group, hkContainerHeapAllocator>") // hkArray< struct hkQemSimplifier::Group, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkQemSimplifier, m_topology, 0, "hkgpVertexTriangleTopology<hkQemSimplifier::Triangle>") // class hkgpVertexTriangleTopology< struct hkQemSimplifier::Triangle >
    HK_TRACKER_MEMBER(hkQemSimplifier, m_positions, 0, "hkArray<hkVector4f, hkContainerHeapAllocator>") // hkArray< hkVector4f, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkQemSimplifier, m_controller, 0, "hkQemSimplifier::ContractionController *") // class hkQemSimplifier::ContractionController *
    HK_TRACKER_MEMBER(hkQemSimplifier, m_scaleCalc, 0, "hkQemSimplifier::ScaleCalculator *") // class hkQemSimplifier::ScaleCalculator *
    HK_TRACKER_MEMBER(hkQemSimplifier, m_materialBoundaries, 0, "hkArray<hkQemSimplifier::BoundaryEdge, hkContainerHeapAllocator>") // hkArray< struct hkQemSimplifier::BoundaryEdge, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkQemSimplifier, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkQemSimplifier, AttributeType, s_libraryName)


// EdgeMap hkQemSimplifier

HK_TRACKER_DECLARE_CLASS_BEGIN(hkQemSimplifier::EdgeMap)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkQemSimplifier::EdgeMap)
    HK_TRACKER_MEMBER(hkQemSimplifier::EdgeMap, m_edgeMap, 0, "hkPointerMap<hkUint64, hkUint32, hkContainerHeapAllocator>") // class hkPointerMap< hkUint64, hkUint32, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkQemSimplifier::EdgeMap, s_libraryName)


// ContractionController hkQemSimplifier

HK_TRACKER_DECLARE_CLASS_BEGIN(hkQemSimplifier::ContractionController)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkQemSimplifier::ContractionController)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkQemSimplifier::ContractionController, s_libraryName, hkReferencedObject)


// ScaleCalculator hkQemSimplifier

HK_TRACKER_DECLARE_CLASS_BEGIN(hkQemSimplifier::ScaleCalculator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkQemSimplifier::ScaleCalculator)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkQemSimplifier::ScaleCalculator, s_libraryName, hkReferencedObject)


// SizeScaleCalculator hkQemSimplifier

HK_TRACKER_DECLARE_CLASS_BEGIN(hkQemSimplifier::SizeScaleCalculator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkQemSimplifier::SizeScaleCalculator)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkQemSimplifier::SizeScaleCalculator, s_libraryName, hkQemSimplifier::ScaleCalculator)


// AngleScaleCalculator hkQemSimplifier

HK_TRACKER_DECLARE_CLASS_BEGIN(hkQemSimplifier::AngleScaleCalculator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkQemSimplifier::AngleScaleCalculator)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkQemSimplifier::AngleScaleCalculator, s_libraryName, hkQemSimplifier::ScaleCalculator)


// AttributeEntry hkQemSimplifier
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkQemSimplifier, AttributeEntry, s_libraryName)


// AttributeFormat hkQemSimplifier

HK_TRACKER_DECLARE_CLASS_BEGIN(hkQemSimplifier::AttributeFormat)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkQemSimplifier::AttributeFormat)
    HK_TRACKER_MEMBER(hkQemSimplifier::AttributeFormat, m_entries, 0, "hkArray<hkQemSimplifier::AttributeEntry, hkContainerHeapAllocator>") // hkArray< struct hkQemSimplifier::AttributeEntry, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkQemSimplifier::AttributeFormat, s_libraryName)


// Group hkQemSimplifier

HK_TRACKER_DECLARE_CLASS_BEGIN(hkQemSimplifier::Group)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkQemSimplifier::Group)
    HK_TRACKER_MEMBER(hkQemSimplifier::Group, m_availableAttributeIndices, 0, "hkArray<hkInt32, hkContainerHeapAllocator>") // hkArray< hkInt32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkQemSimplifier::Group, m_attributes, 0, "hkArray<hkVector4f*, hkContainerHeapAllocator>") // hkArray< hkVector4f*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkQemSimplifier::Group, m_attributeFreeList, 0, "hkFreeList") // class hkFreeList
    HK_TRACKER_MEMBER(hkQemSimplifier::Group, m_fmt, 0, "hkQemSimplifier::AttributeFormat") // struct hkQemSimplifier::AttributeFormat
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkQemSimplifier::Group, s_libraryName)


// EdgeContraction hkQemSimplifier
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkQemSimplifier, EdgeContraction, s_libraryName)


// Triangle hkQemSimplifier
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkQemSimplifier, Triangle, s_libraryName)


// Attribute hkQemSimplifier
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkQemSimplifier, Attribute, s_libraryName)


// BoundaryEdge hkQemSimplifier
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkQemSimplifier, BoundaryEdge, s_libraryName)

#include <Common/GeometryUtilities/Mesh/Simplifiers/QemSimplifier/hkQemVertexContractionController.h>


// hkQemVertexContractionController ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkQemVertexContractionController)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkQemVertexContractionController)
    HK_TRACKER_MEMBER(hkQemVertexContractionController, m_bitField, 0, "hkArray<hkUint32, hkContainerHeapAllocator>") // hkArray< hkUint32, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkQemVertexContractionController, s_libraryName, hkQemSimplifier::ContractionController)

#include <Common/GeometryUtilities/Mesh/Simplifiers/QemSimplifier/hkQuadricMetric.h>


// hkQuadricMetric ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkQuadricMetric)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkQuadricMetric)
    HK_TRACKER_MEMBER(hkQuadricMetric, m_a, 0, "hkMatrixfNm") // class hkMatrixfNm
    HK_TRACKER_MEMBER(hkQuadricMetric, m_b, 0, "hkVectorNf") // class hkVectorNf
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkQuadricMetric, s_libraryName)

#include <Common/GeometryUtilities/Mesh/Skin/hkSkinBinding.h>


// hkSkinBinding ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkSkinBinding)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkSkinBinding)
    HK_TRACKER_MEMBER(hkSkinBinding, m_skin, 0, "hkMeshShape *") // class hkMeshShape *
    HK_TRACKER_MEMBER(hkSkinBinding, m_worldFromBoneTransforms, 0, "hkArray<hkMatrix4f, hkContainerHeapAllocator>") // hkArray< hkMatrix4f, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkSkinBinding, m_boneNames, 0, "hkArray<hkStringPtr, hkContainerHeapAllocator>") // hkArray< hkStringPtr, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkSkinBinding, s_libraryName, hkMeshShape)

#include <Common/GeometryUtilities/Mesh/Skin/hkSkinnedMeshShape.h>


// hkSkinnedMeshShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkSkinnedMeshShape)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BoneSetIdDiscriminant)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BoneSet)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BoneSection)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Part)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkSkinnedMeshShape)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkSkinnedMeshShape, s_libraryName, hkReferencedObject)


// BoneSetIdDiscriminant hkSkinnedMeshShape
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkSkinnedMeshShape, BoneSetIdDiscriminant, s_libraryName)


// BoneSet hkSkinnedMeshShape
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkSkinnedMeshShape, BoneSet, s_libraryName)


// BoneSection hkSkinnedMeshShape

HK_TRACKER_DECLARE_CLASS_BEGIN(hkSkinnedMeshShape::BoneSection)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkSkinnedMeshShape::BoneSection)
    HK_TRACKER_MEMBER(hkSkinnedMeshShape::BoneSection, m_meshBuffer, 0, "hkMeshShape *") // class hkMeshShape *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkSkinnedMeshShape::BoneSection, s_libraryName)


// Part hkSkinnedMeshShape
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkSkinnedMeshShape, Part, s_libraryName)


// hkStorageSkinnedMeshShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkStorageSkinnedMeshShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkStorageSkinnedMeshShape)
    HK_TRACKER_MEMBER(hkStorageSkinnedMeshShape, m_bonesBuffer, 0, "hkArray<hkInt16, hkContainerHeapAllocator>") // hkArray< hkInt16, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkStorageSkinnedMeshShape, m_boneSets, 0, "hkArray<hkSkinnedMeshShape::BoneSet, hkContainerHeapAllocator>") // hkArray< struct hkSkinnedMeshShape::BoneSet, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkStorageSkinnedMeshShape, m_boneSections, 0, "hkArray<hkSkinnedMeshShape::BoneSection, hkContainerHeapAllocator>") // hkArray< struct hkSkinnedMeshShape::BoneSection, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkStorageSkinnedMeshShape, m_parts, 0, "hkArray<hkSkinnedMeshShape::Part, hkContainerHeapAllocator>") // hkArray< struct hkSkinnedMeshShape::Part, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkStorageSkinnedMeshShape, m_name, 0, "hkStringPtr") // hkStringPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkStorageSkinnedMeshShape, s_libraryName, hkSkinnedMeshShape)

#include <Common/GeometryUtilities/Mesh/Skin/hkSkinnedMeshShapeBuilder.h>


// hkSkinnedMeshBuilder ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkSkinnedMeshBuilder)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SkinningInfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(VertexBuffer)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MeshSection)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SkinDescriptor)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkSkinnedMeshBuilder)
    HK_TRACKER_MEMBER(hkSkinnedMeshBuilder, m_surfaces, 0, "hkArray<hkMeshMaterial*, hkContainerHeapAllocator>") // hkArray< class hkMeshMaterial*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkSkinnedMeshBuilder, m_sourceMeshes, 0, "hkArray<hkMeshShape *, hkContainerHeapAllocator>") // hkArray< const class hkMeshShape *, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkSkinnedMeshBuilder, m_sourceMeshTransforms, 0, "hkArray<hkQTransformf, hkContainerHeapAllocator>") // hkArray< class hkQTransformf, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkSkinnedMeshBuilder, m_sourceMeshSections, 0, "hkArray<hkSkinnedMeshBuilder::MeshSection, hkContainerHeapAllocator>") // hkArray< struct hkSkinnedMeshBuilder::MeshSection, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkSkinnedMeshBuilder, m_boneSets, 0, "hkArray<hkMeshBoneIndexMapping, hkContainerHeapAllocator>") // hkArray< struct hkMeshBoneIndexMapping, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkSkinnedMeshBuilder, m_skinDescriptors, 0, "hkArray<hkSkinnedMeshBuilder::SkinDescriptor, hkContainerHeapAllocator>") // hkArray< struct hkSkinnedMeshBuilder::SkinDescriptor, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkSkinnedMeshBuilder, m_skinnedMeshShape, 0, "hkSkinnedMeshShape*") // class hkSkinnedMeshShape*
    HK_TRACKER_MEMBER(hkSkinnedMeshBuilder, m_meshSystem, 0, "hkMeshSystem*") // class hkMeshSystem*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkSkinnedMeshBuilder, s_libraryName)


// SkinningInfo hkSkinnedMeshBuilder
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkSkinnedMeshBuilder, SkinningInfo, s_libraryName)


// VertexBuffer hkSkinnedMeshBuilder

HK_TRACKER_DECLARE_CLASS_BEGIN(hkSkinnedMeshBuilder::VertexBuffer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkSkinnedMeshBuilder::VertexBuffer)
    HK_TRACKER_MEMBER(hkSkinnedMeshBuilder::VertexBuffer, m_vb, 0, "hkMeshVertexBuffer *") // class hkMeshVertexBuffer *
    HK_TRACKER_MEMBER(hkSkinnedMeshBuilder::VertexBuffer, m_dominants, 0, "hkDisplacementMappingUtil::DominantsBuffer") // class hkDisplacementMappingUtil::DominantsBuffer
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkSkinnedMeshBuilder::VertexBuffer, s_libraryName, hkReferencedObject)


// MeshSection hkSkinnedMeshBuilder

HK_TRACKER_DECLARE_CLASS_BEGIN(hkSkinnedMeshBuilder::MeshSection)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkSkinnedMeshBuilder::MeshSection)
    HK_TRACKER_MEMBER(hkSkinnedMeshBuilder::MeshSection, m_originalUsedVertices, 0, "hkBitField") // class hkBitField
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkSkinnedMeshBuilder::MeshSection, s_libraryName)


// SkinDescriptor hkSkinnedMeshBuilder

HK_TRACKER_DECLARE_CLASS_BEGIN(hkSkinnedMeshBuilder::SkinDescriptor)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkSkinnedMeshBuilder::SkinDescriptor)
    HK_TRACKER_MEMBER(hkSkinnedMeshBuilder::SkinDescriptor, m_localFromWorldBoneMap, 0, "hkMeshBoneIndexMapping") // struct hkMeshBoneIndexMapping
    HK_TRACKER_MEMBER(hkSkinnedMeshBuilder::SkinDescriptor, m_worldFromLocalBoneMap, 0, "hkMeshBoneIndexMapping") // struct hkMeshBoneIndexMapping
    HK_TRACKER_MEMBER(hkSkinnedMeshBuilder::SkinDescriptor, m_usedBones, 0, "hkBitField") // class hkBitField
    HK_TRACKER_MEMBER(hkSkinnedMeshBuilder::SkinDescriptor, m_sections, 0, "hkArray<hkSkinnedMeshBuilder::MeshSection, hkContainerHeapAllocator>") // hkArray< struct hkSkinnedMeshBuilder::MeshSection, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkSkinnedMeshBuilder::SkinDescriptor, s_libraryName)

#include <Common/GeometryUtilities/Mesh/Skin/hkSkinnedRefMeshShape.h>


// hkSkinnedRefMeshShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkSkinnedRefMeshShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkSkinnedRefMeshShape)
    HK_TRACKER_MEMBER(hkSkinnedRefMeshShape, m_skinnedMeshShape, 0, "hkSkinnedMeshShape *") // class hkSkinnedMeshShape *
    HK_TRACKER_MEMBER(hkSkinnedRefMeshShape, m_bones, 0, "hkArray<hkInt16, hkContainerHeapAllocator>") // hkArray< hkInt16, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkSkinnedRefMeshShape, m_localFromRootTransforms, 0, "hkArray<hkVector4f, hkContainerHeapAllocator>") // hkArray< hkVector4f, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkSkinnedRefMeshShape, m_name, 0, "hkStringPtr") // hkStringPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkSkinnedRefMeshShape, s_libraryName, hkMeshShape)

#include <Common/GeometryUtilities/Mesh/Utils/BarycentricVertexInterpolator/hkBarycentricVertexInterpolator.h>


// hkBarycentricVertexInterpolator ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkBarycentricVertexInterpolator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkBarycentricVertexInterpolator)
    HK_TRACKER_MEMBER(hkBarycentricVertexInterpolator, m_srcLockedVertices, 0, "hkMeshVertexBuffer::LockedVertices") // struct hkMeshVertexBuffer::LockedVertices
    HK_TRACKER_MEMBER(hkBarycentricVertexInterpolator, m_srcPositionBuffer, 0, "hkMeshVertexBuffer::LockedVertices::Buffer") // struct hkMeshVertexBuffer::LockedVertices::Buffer
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkBarycentricVertexInterpolator, s_libraryName)

#include <Common/GeometryUtilities/Mesh/Utils/DisplacementMappingUtil/hkDisplacementMappingUtil.h>


// hkDisplacementMappingUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDisplacementMappingUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(DominantInfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(DominantsBuffer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkDisplacementMappingUtil, s_libraryName)


// DominantInfo hkDisplacementMappingUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkDisplacementMappingUtil, DominantInfo, s_libraryName)


// DominantsBuffer hkDisplacementMappingUtil

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDisplacementMappingUtil::DominantsBuffer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDisplacementMappingUtil::DominantsBuffer)
    HK_TRACKER_MEMBER(hkDisplacementMappingUtil::DominantsBuffer, m_data, 0, "hkUint8*") // hkUint8*
    HK_TRACKER_MEMBER(hkDisplacementMappingUtil::DominantsBuffer, m_texture, 0, "hkMeshTexture *") // class hkMeshTexture *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkDisplacementMappingUtil::DominantsBuffer, s_libraryName)

#include <Common/GeometryUtilities/Mesh/Utils/FindClosestPositionUtil/hkFindClosestPositionUtil.h>


// hkFindClosestPositionUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkFindClosestPositionUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(IntCoord)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Box)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkFindClosestPositionUtil)
    HK_TRACKER_MEMBER(hkFindClosestPositionUtil, m_positions, 0, "hkArray<hkVector4f, hkContainerHeapAllocator>") // hkArray< hkVector4f, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkFindClosestPositionUtil, m_boxFreeList, 0, "hkFreeList") // class hkFreeList
    HK_TRACKER_MEMBER(hkFindClosestPositionUtil, m_hashMap, 0, "hkPointerMap<hkUint32, hkFindClosestPositionUtil::Box*, hkContainerHeapAllocator>") // class hkPointerMap< hkUint32, struct hkFindClosestPositionUtil::Box*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkFindClosestPositionUtil, s_libraryName)


// IntCoord hkFindClosestPositionUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkFindClosestPositionUtil, IntCoord, s_libraryName)


// Box hkFindClosestPositionUtil

HK_TRACKER_DECLARE_CLASS_BEGIN(hkFindClosestPositionUtil::Box)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkFindClosestPositionUtil::Box)
    HK_TRACKER_MEMBER(hkFindClosestPositionUtil::Box, m_next, 0, "hkFindClosestPositionUtil::Box*") // struct hkFindClosestPositionUtil::Box*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkFindClosestPositionUtil::Box, s_libraryName)

#include <Common/GeometryUtilities/Mesh/Utils/FindUniqueIndicesUtil/hkFindUniqueIndicesUtil.h>


// hkFindUniqueIndicesUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkFindUniqueIndicesUtil)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkFindUniqueIndicesUtil)
    HK_TRACKER_MEMBER(hkFindUniqueIndicesUtil, m_indicesMap, 0, "hkArray<hkInt32, hkContainerHeapAllocator>") // hkArray< hkInt32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkFindUniqueIndicesUtil, m_uniqueIndices, 0, "hkArray<hkInt32, hkContainerHeapAllocator>") // hkArray< hkInt32, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkFindUniqueIndicesUtil, s_libraryName)

#include <Common/GeometryUtilities/Mesh/Utils/FindUniquePositionsUtil/hkFindUniquePositionsUtil.h>


// hkFindUniquePositionsUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkFindUniquePositionsUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Entry)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkFindUniquePositionsUtil)
    HK_TRACKER_MEMBER(hkFindUniquePositionsUtil, m_positions, 0, "hkArray<hkVector4f, hkContainerHeapAllocator>") // hkArray< hkVector4f, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkFindUniquePositionsUtil, m_hashMap, 0, "hkPointerMap<hkUint32, hkInt32, hkContainerHeapAllocator>") // class hkPointerMap< hkUint32, hkInt32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkFindUniquePositionsUtil, m_entries, 0, "hkArray<hkFindUniquePositionsUtil::Entry, hkContainerHeapAllocator>") // hkArray< struct hkFindUniquePositionsUtil::Entry, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkFindUniquePositionsUtil, s_libraryName)


// Entry hkFindUniquePositionsUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkFindUniquePositionsUtil, Entry, s_libraryName)

#include <Common/GeometryUtilities/Mesh/Utils/FindVertexWeightsUtil/hkFindVertexWeightsUtil.h>


// hkFindVertexWeightsUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkFindVertexWeightsUtil, s_libraryName)

#include <Common/GeometryUtilities/Mesh/Utils/IndexSet/hkIndexSet.h>


// hkIndexSet ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkIndexSet)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkIndexSet)
    HK_TRACKER_MEMBER(hkIndexSet, m_indices, 0, "hkArray<hkInt32, hkContainerHeapAllocator>") // hkArray< hkInt32, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkIndexSet, s_libraryName)

#include <Common/GeometryUtilities/Mesh/Utils/LinearVertexCalculator/hkLinearVertexCalculator.h>


// hkLinearVertexCalculator ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkLinearVertexCalculator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkLinearVertexCalculator)
    HK_TRACKER_MEMBER(hkLinearVertexCalculator, m_vertexBuffer, 0, "hkMeshVertexBuffer*") // class hkMeshVertexBuffer*
    HK_TRACKER_MEMBER(hkLinearVertexCalculator, m_lockedVertices, 0, "hkMeshVertexBuffer::LockedVertices") // struct hkMeshVertexBuffer::LockedVertices
    HK_TRACKER_MEMBER(hkLinearVertexCalculator, m_rootTriangleValues, 0, "hkArray<hkVector4f, hkContainerHeapAllocator>") // hkArray< hkVector4f, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkLinearVertexCalculator, s_libraryName, hkReferencedObject)

#include <Common/GeometryUtilities/Mesh/Utils/MeshMaterialUtil/hkMeshMaterialUtil.h>


// hkMeshMaterialUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkMeshMaterialUtil, s_libraryName)

#include <Common/GeometryUtilities/Mesh/Utils/MeshSectionBuilder/hkMeshSectionBuilder.h>


// hkMeshSectionBuilder ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMeshSectionBuilder)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMeshSectionBuilder)
    HK_TRACKER_MEMBER(hkMeshSectionBuilder, m_sections, 0, "hkArray<hkMeshSectionCinfo, hkContainerHeapAllocator>") // hkArray< struct hkMeshSectionCinfo, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkMeshSectionBuilder, m_indices16, 0, "hkArray<hkUint16, hkContainerHeapAllocator>") // hkArray< hkUint16, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkMeshSectionBuilder, m_indices32, 0, "hkArray<hkUint32, hkContainerHeapAllocator>") // hkArray< hkUint32, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkMeshSectionBuilder, s_libraryName)

#include <Common/GeometryUtilities/Mesh/Utils/MeshSectionLockSet/hkMeshSectionLockSet.h>


// hkMeshSectionLockSet ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMeshSectionLockSet)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMeshSectionLockSet)
    HK_TRACKER_MEMBER(hkMeshSectionLockSet, m_sections, 0, "hkArray<hkMeshSection, hkContainerHeapAllocator>") // hkArray< struct hkMeshSection, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkMeshSectionLockSet, m_shapes, 0, "hkArray<hkMeshShape*, hkContainerHeapAllocator>") // hkArray< const class hkMeshShape*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkMeshSectionLockSet, s_libraryName)

#include <Common/GeometryUtilities/Mesh/Utils/MeshSectionMergeUtil/hkMeshSectionMergeUtil.h>


// hkMeshSectionMergeUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkMeshSectionMergeUtil, s_libraryName)

#include <Common/GeometryUtilities/Mesh/Utils/MeshShapeUtil/hkMeshShapeUtil.h>


// hkMeshShapeUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMeshShapeUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Statistics)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkMeshShapeUtil, s_libraryName)


// Statistics hkMeshShapeUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkMeshShapeUtil, Statistics, s_libraryName)

#include <Common/GeometryUtilities/Mesh/Utils/NormalCalculator/hkNormalCalculator.h>


// hkNormalCalculator ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkNormalCalculator, s_libraryName)

#include <Common/GeometryUtilities/Mesh/Utils/PrimitiveUtil/hkMeshPrimitiveUtil.h>


// hkMeshPrimitiveUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMeshPrimitiveUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(PrimitiveProvider)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(PrimitiveStyle)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkMeshPrimitiveUtil, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkMeshPrimitiveUtil, PrimitiveStyle, s_libraryName)


// PrimitiveProvider hkMeshPrimitiveUtil

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMeshPrimitiveUtil::PrimitiveProvider)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMeshPrimitiveUtil::PrimitiveProvider)
    HK_TRACKER_MEMBER(hkMeshPrimitiveUtil::PrimitiveProvider, m_ptr, 0, "hkUint8*") // hkUint8*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkMeshPrimitiveUtil::PrimitiveProvider, s_libraryName)


// hkMergeMeshPrimitvesCalculator ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkMergeMeshPrimitvesCalculator, s_libraryName)

#include <Common/GeometryUtilities/Mesh/Utils/SkinningUtil/hkSkinningUtil.h>


// hkSkinningUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkSkinningUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Entry)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkSkinningUtil, s_libraryName)


// Entry hkSkinningUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkSkinningUtil, Entry, s_libraryName)

#include <Common/GeometryUtilities/Mesh/Utils/VertexBufferSharingUtil/hkVertexBufferSharingUtil.h>


// hkVertexBufferSharingUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkVertexBufferSharingUtil, s_libraryName)

#include <Common/GeometryUtilities/Mesh/Utils/VertexBufferUtil/hkMeshVertexBufferUtil.h>


// hkMeshVertexBufferUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMeshVertexBufferUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Thresholds)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(TransformFlag)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkMeshVertexBufferUtil, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkMeshVertexBufferUtil, TransformFlag, s_libraryName)


// Thresholds hkMeshVertexBufferUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkMeshVertexBufferUtil, Thresholds, s_libraryName)

#include <Common/GeometryUtilities/Mesh/Utils/VertexFloat32Converter/hkVertexFloat32Converter.h>


// hkVertexFloat32Converter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkVertexFloat32Converter)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Entry)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SrcType)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(DstType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkVertexFloat32Converter)
    HK_TRACKER_MEMBER(hkVertexFloat32Converter, m_entries, 0, "hkArray<hkVertexFloat32Converter::Entry, hkContainerHeapAllocator>") // hkArray< struct hkVertexFloat32Converter::Entry, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkVertexFloat32Converter, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkVertexFloat32Converter, SrcType, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkVertexFloat32Converter, DstType, s_libraryName)


// Entry hkVertexFloat32Converter
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkVertexFloat32Converter, Entry, s_libraryName)

#include <Common/GeometryUtilities/Mesh/Utils/VertexSharingUtil/hkVertexSharingUtil.h>


// hkVertexSharingUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkVertexSharingUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Threshold)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Entry)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkVertexSharingUtil)
    HK_TRACKER_MEMBER(hkVertexSharingUtil, m_hashMap, 0, "hkPointerMap<hkUint32, hkInt32, hkContainerHeapAllocator>") // class hkPointerMap< hkUint32, hkInt32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkVertexSharingUtil, m_entries, 0, "hkArray<hkVertexSharingUtil::Entry, hkContainerHeapAllocator>") // hkArray< struct hkVertexSharingUtil::Entry, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkVertexSharingUtil, m_vertices, 0, "hkArray<hkUint8, hkContainerHeapAllocator>") // hkArray< hkUint8, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkVertexSharingUtil, m_workVertex, 0, "hkArray<hkUint8, hkContainerHeapAllocator>") // hkArray< hkUint8, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkVertexSharingUtil, m_lockedWorkVertex, 0, "hkMeshVertexBuffer::LockedVertices") // struct hkMeshVertexBuffer::LockedVertices
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkVertexSharingUtil, s_libraryName)


// Threshold hkVertexSharingUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkVertexSharingUtil, Threshold, s_libraryName)


// Entry hkVertexSharingUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkVertexSharingUtil, Entry, s_libraryName)

#include <Common/GeometryUtilities/Mesh/hkMeshBody.h>


// hkMeshBody ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMeshBody)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(PickDataIdentifier)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMeshBody)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkMeshBody, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkMeshBody, PickDataIdentifier, s_libraryName)

#include <Common/GeometryUtilities/Mesh/hkMeshMaterialRegistry.h>


// hkMeshMaterialRegistry ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMeshMaterialRegistry)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMeshMaterialRegistry)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkMeshMaterialRegistry, s_libraryName, hkReferencedObject)

#include <Common/GeometryUtilities/Mesh/hkMeshShape.h>


// hkMeshMaterial ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMeshMaterial)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMeshMaterial)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkMeshMaterial, s_libraryName, hkReferencedObject)


// hkMeshSection ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMeshSection)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MeshSectionIndexType)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(PrimitiveType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMeshSection)
    HK_TRACKER_MEMBER(hkMeshSection, m_indices, 0, "void*") // const void*
    HK_TRACKER_MEMBER(hkMeshSection, m_vertexBuffer, 0, "hkMeshVertexBuffer*") // class hkMeshVertexBuffer*
    HK_TRACKER_MEMBER(hkMeshSection, m_material, 0, "hkMeshMaterial*") // class hkMeshMaterial*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkMeshSection, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkMeshSection, MeshSectionIndexType, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkMeshSection, PrimitiveType, s_libraryName)


// hkMeshSectionCinfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMeshSectionCinfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMeshSectionCinfo)
    HK_TRACKER_MEMBER(hkMeshSectionCinfo, m_vertexBuffer, 0, "hkMeshVertexBuffer*") // class hkMeshVertexBuffer*
    HK_TRACKER_MEMBER(hkMeshSectionCinfo, m_material, 0, "hkMeshMaterial*") // class hkMeshMaterial*
    HK_TRACKER_MEMBER(hkMeshSectionCinfo, m_indices, 0, "void*") // const void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkMeshSectionCinfo, s_libraryName)


// hkMeshShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMeshShape)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(AccessFlags)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMeshShape)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkMeshShape, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkMeshShape, AccessFlags, s_libraryName)

#include <Common/GeometryUtilities/Mesh/hkMeshSystem.h>


// hkMeshSystem ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMeshSystem)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMeshSystem)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkMeshSystem, s_libraryName, hkReferencedObject)

#include <Common/GeometryUtilities/Mesh/hkMeshTexture.h>


// hkMeshTexture ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMeshTexture)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Sampler)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(RawBufferDescriptor)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Format)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(FilterMode)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(TextureUsageType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMeshTexture)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkMeshTexture, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkMeshTexture, Format, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkMeshTexture, FilterMode, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkMeshTexture, TextureUsageType, s_libraryName)


// Sampler hkMeshTexture

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMeshTexture::Sampler)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMeshTexture::Sampler)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkMeshTexture::Sampler, s_libraryName, hkReferencedObject)


// RawBufferDescriptor hkMeshTexture
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkMeshTexture, RawBufferDescriptor, s_libraryName)

#include <Common/GeometryUtilities/Mesh/hkMeshVertexBuffer.h>


// hkVertexFormat ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkVertexFormat)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Element)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ComponentType)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ComponentUsage)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(HintFlags)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SharingType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkVertexFormat, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkVertexFormat, ComponentType, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkVertexFormat, ComponentUsage, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkVertexFormat, HintFlags, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkVertexFormat, SharingType, s_libraryName)


// Element hkVertexFormat
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkVertexFormat, Element, s_libraryName)


// hkMeshVertexBuffer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMeshVertexBuffer)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(LockedVertices)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(LockInput)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(PartialLockInput)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Flags)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(LockResult)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMeshVertexBuffer)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkMeshVertexBuffer, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkMeshVertexBuffer, Flags, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkMeshVertexBuffer, LockResult, s_libraryName)


// LockedVertices hkMeshVertexBuffer

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMeshVertexBuffer::LockedVertices)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Buffer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMeshVertexBuffer::LockedVertices)
    HK_TRACKER_MEMBER(hkMeshVertexBuffer::LockedVertices, m_buffers, 0, "hkMeshVertexBuffer::LockedVertices::Buffer [32]") // struct hkMeshVertexBuffer::LockedVertices::Buffer [32]
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkMeshVertexBuffer::LockedVertices, s_libraryName)


// Buffer hkMeshVertexBuffer::LockedVertices

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMeshVertexBuffer::LockedVertices::Buffer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMeshVertexBuffer::LockedVertices::Buffer)
    HK_TRACKER_MEMBER(hkMeshVertexBuffer::LockedVertices::Buffer, m_start, 0, "void*") // void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkMeshVertexBuffer::LockedVertices::Buffer, s_libraryName)


// LockInput hkMeshVertexBuffer
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkMeshVertexBuffer, LockInput, s_libraryName)


// PartialLockInput hkMeshVertexBuffer
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkMeshVertexBuffer, PartialLockInput, s_libraryName)

#include <Common/GeometryUtilities/Misc/hkGeometryUtils.h>


// hkGeometryUtils ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkGeometryUtils)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(IVertices)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Node)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(GetGeometryInsideAabbMode)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkGeometryUtils, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkGeometryUtils, GetGeometryInsideAabbMode, s_libraryName)


// IVertices hkGeometryUtils

HK_TRACKER_DECLARE_CLASS_BEGIN(hkGeometryUtils::IVertices)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkGeometryUtils::IVertices)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkGeometryUtils::IVertices, s_libraryName)


// Node hkGeometryUtils

HK_TRACKER_DECLARE_CLASS_BEGIN(hkGeometryUtils::Node)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Triangle)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Edge)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkGeometryUtils::Node)
    HK_TRACKER_MEMBER(hkGeometryUtils::Node, m_edges, 0, "hkArray<hkGeometryUtils::Node::Edge, hkContainerHeapAllocator>") // hkArray< struct hkGeometryUtils::Node::Edge, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkGeometryUtils::Node, s_libraryName)


// Triangle hkGeometryUtils::Node
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkGeometryUtils::Node, Triangle, s_libraryName)


// Edge hkGeometryUtils::Node

HK_TRACKER_DECLARE_CLASS_BEGIN(hkGeometryUtils::Node::Edge)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkGeometryUtils::Node::Edge)
    HK_TRACKER_MEMBER(hkGeometryUtils::Node::Edge, m_triangles, 0, "hkArray<hkGeometryUtils::Node::Triangle, hkContainerHeapAllocator>") // hkArray< struct hkGeometryUtils::Node::Triangle, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkGeometryUtils::Node::Edge, m_triangleIndices, 0, "hkArray<hkUint32, hkContainerHeapAllocator>") // hkArray< hkUint32, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkGeometryUtils::Node::Edge, s_libraryName)

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
