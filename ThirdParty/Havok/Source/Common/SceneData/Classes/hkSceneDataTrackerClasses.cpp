/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#include <Common/SceneData/hkSceneData.h>
static const char s_libraryName[] = "hkSceneData";
#include <Common/Base/Memory/Tracker/hkTrackerClassDefinition.h>

void HK_CALL hkSceneDataRegister() {}

#include <Common/SceneData/AlignScene/hkAlignSceneToNodeOptions.h>


// hkAlignSceneToNodeOptions ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkAlignSceneToNodeOptions)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkAlignSceneToNodeOptions)
    HK_TRACKER_MEMBER(hkAlignSceneToNodeOptions, m_nodeName, 0, "hkStringPtr") // hkStringPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkAlignSceneToNodeOptions, s_libraryName, hkReferencedObject)

#include <Common/SceneData/Attributes/hkxAnimatedFloat.h>


// hkxAnimatedFloat ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxAnimatedFloat)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxAnimatedFloat)
    HK_TRACKER_MEMBER(hkxAnimatedFloat, m_floats, 0, "hkArray<float, hkContainerHeapAllocator>") // hkArray< float, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxAnimatedFloat, s_libraryName, hkReferencedObject)

#include <Common/SceneData/Attributes/hkxAnimatedMatrix.h>


// hkxAnimatedMatrix ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxAnimatedMatrix)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxAnimatedMatrix)
    HK_TRACKER_MEMBER(hkxAnimatedMatrix, m_matrices, 0, "hkArray<float, hkContainerHeapAllocator>") // hkArray< float, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxAnimatedMatrix, s_libraryName, hkReferencedObject)

#include <Common/SceneData/Attributes/hkxAnimatedQuaternion.h>


// hkxAnimatedQuaternion ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxAnimatedQuaternion)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxAnimatedQuaternion)
    HK_TRACKER_MEMBER(hkxAnimatedQuaternion, m_quaternions, 0, "hkArray<float, hkContainerHeapAllocator>") // hkArray< float, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxAnimatedQuaternion, s_libraryName, hkReferencedObject)

#include <Common/SceneData/Attributes/hkxAnimatedVector.h>


// hkxAnimatedVector ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxAnimatedVector)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxAnimatedVector)
    HK_TRACKER_MEMBER(hkxAnimatedVector, m_vectors, 0, "hkArray<float, hkContainerHeapAllocator>") // hkArray< float, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxAnimatedVector, s_libraryName, hkReferencedObject)

#include <Common/SceneData/Attributes/hkxAttribute.h>


// hkxAttribute ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxAttribute)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Hint)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxAttribute)
    HK_TRACKER_MEMBER(hkxAttribute, m_name, 0, "hkStringPtr") // hkStringPtr
    HK_TRACKER_MEMBER(hkxAttribute, m_value, 0, "hkReferencedObject *") // class hkReferencedObject *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkxAttribute, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkxAttribute, Hint, s_libraryName)

#include <Common/SceneData/Attributes/hkxAttributeGroup.h>


// hkxAttributeGroup ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxAttributeGroup)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxAttributeGroup)
    HK_TRACKER_MEMBER(hkxAttributeGroup, m_name, 0, "hkStringPtr") // hkStringPtr
    HK_TRACKER_MEMBER(hkxAttributeGroup, m_attributes, 0, "hkArray<hkxAttribute, hkContainerHeapAllocator>") // hkArray< struct hkxAttribute, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkxAttributeGroup, s_libraryName)

#include <Common/SceneData/Attributes/hkxAttributeHolder.h>


// hkxAttributeHolder ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxAttributeHolder)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxAttributeHolder)
    HK_TRACKER_MEMBER(hkxAttributeHolder, m_attributeGroups, 0, "hkArray<hkxAttributeGroup, hkContainerHeapAllocator>") // hkArray< struct hkxAttributeGroup, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxAttributeHolder, s_libraryName, hkReferencedObject)

#include <Common/SceneData/Attributes/hkxSparselyAnimatedBool.h>


// hkxSparselyAnimatedBool ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxSparselyAnimatedBool)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxSparselyAnimatedBool)
    HK_TRACKER_MEMBER(hkxSparselyAnimatedBool, m_bools, 0, "hkArray<hkBool, hkContainerHeapAllocator>") // hkArray< hkBool, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkxSparselyAnimatedBool, m_times, 0, "hkArray<float, hkContainerHeapAllocator>") // hkArray< float, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxSparselyAnimatedBool, s_libraryName, hkReferencedObject)

#include <Common/SceneData/Attributes/hkxSparselyAnimatedEnum.h>


// hkxEnum ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxEnum)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Item)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxEnum)
    HK_TRACKER_MEMBER(hkxEnum, m_items, 0, "hkArray<hkxEnum::Item, hkContainerHeapAllocator>") // hkArray< struct hkxEnum::Item, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxEnum, s_libraryName, hkReferencedObject)


// Item hkxEnum

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxEnum::Item)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxEnum::Item)
    HK_TRACKER_MEMBER(hkxEnum::Item, m_name, 0, "hkStringPtr") // hkStringPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkxEnum::Item, s_libraryName)


// hkxSparselyAnimatedEnum ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxSparselyAnimatedEnum)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxSparselyAnimatedEnum)
    HK_TRACKER_MEMBER(hkxSparselyAnimatedEnum, m_enum, 0, "hkxEnum *") // class hkxEnum *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxSparselyAnimatedEnum, s_libraryName, hkxSparselyAnimatedInt)

#include <Common/SceneData/Attributes/hkxSparselyAnimatedInt.h>


// hkxSparselyAnimatedInt ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxSparselyAnimatedInt)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxSparselyAnimatedInt)
    HK_TRACKER_MEMBER(hkxSparselyAnimatedInt, m_ints, 0, "hkArray<hkInt32, hkContainerHeapAllocator>") // hkArray< hkInt32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkxSparselyAnimatedInt, m_times, 0, "hkArray<float, hkContainerHeapAllocator>") // hkArray< float, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxSparselyAnimatedInt, s_libraryName, hkReferencedObject)

#include <Common/SceneData/Attributes/hkxSparselyAnimatedString.h>


// hkxSparselyAnimatedString ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxSparselyAnimatedString)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxSparselyAnimatedString)
    HK_TRACKER_MEMBER(hkxSparselyAnimatedString, m_strings, 0, "hkArray<hkStringPtr, hkContainerHeapAllocator>") // hkArray< hkStringPtr, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkxSparselyAnimatedString, m_times, 0, "hkArray<float, hkContainerHeapAllocator>") // hkArray< float, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxSparselyAnimatedString, s_libraryName, hkReferencedObject)

#include <Common/SceneData/Blob/hkxBlob.h>


// hkxBlob ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxBlob)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxBlob)
    HK_TRACKER_MEMBER(hkxBlob, m_data, 0, "hkArray<hkUint8, hkContainerHeapAllocator>") // hkArray< hkUint8, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxBlob, s_libraryName, hkReferencedObject)

#include <Common/SceneData/Blob/hkxBlobMeshShape.h>


// hkxBlobMeshShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxBlobMeshShape)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxBlobMeshShape)
    HK_TRACKER_MEMBER(hkxBlobMeshShape, m_blob, 0, "hkxBlob") // class hkxBlob
    HK_TRACKER_MEMBER(hkxBlobMeshShape, m_name, 0, "hkStringPtr") // hkStringPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxBlobMeshShape, s_libraryName, hkMeshShape)

#include <Common/SceneData/Camera/hkxCamera.h>


// hkxCamera ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxCamera)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxCamera)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxCamera, s_libraryName, hkReferencedObject)

#include <Common/SceneData/Environment/hkxEnvironment.h>


// hkxEnvironment ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxEnvironment)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Variable)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxEnvironment)
    HK_TRACKER_MEMBER(hkxEnvironment, m_variables, 0, "hkArray<hkxEnvironment::Variable, hkContainerHeapAllocator>") // hkArray< struct hkxEnvironment::Variable, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxEnvironment, s_libraryName, hkReferencedObject)


// Variable hkxEnvironment

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxEnvironment::Variable)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxEnvironment::Variable)
    HK_TRACKER_MEMBER(hkxEnvironment::Variable, m_name, 0, "hkStringPtr") // hkStringPtr
    HK_TRACKER_MEMBER(hkxEnvironment::Variable, m_value, 0, "hkStringPtr") // hkStringPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkxEnvironment::Variable, s_libraryName)

#include <Common/SceneData/Graph/hkxNode.h>


// hkxNode ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxNode)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(AnnotationData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxNode)
    HK_TRACKER_MEMBER(hkxNode, m_name, 0, "hkStringPtr") // hkStringPtr
    HK_TRACKER_MEMBER(hkxNode, m_object, 0, "hkReferencedObject *") // class hkReferencedObject *
    HK_TRACKER_MEMBER(hkxNode, m_keyFrames, 0, "hkArray<hkMatrix4f, hkContainerHeapAllocator>") // hkArray< hkMatrix4f, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkxNode, m_children, 0, "hkArray<hkxNode *, hkContainerHeapAllocator>") // hkArray< class hkxNode *, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkxNode, m_annotations, 0, "hkArray<hkxNode::AnnotationData, hkContainerHeapAllocator>") // hkArray< struct hkxNode::AnnotationData, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkxNode, m_linearKeyFrameHints, 0, "hkArray<float, hkContainerHeapAllocator>") // hkArray< float, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkxNode, m_userProperties, 0, "hkStringPtr") // hkStringPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxNode, s_libraryName, hkxAttributeHolder)


// AnnotationData hkxNode

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxNode::AnnotationData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxNode::AnnotationData)
    HK_TRACKER_MEMBER(hkxNode::AnnotationData, m_description, 0, "hkStringPtr") // hkStringPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkxNode::AnnotationData, s_libraryName)

#include <Common/SceneData/Light/hkxLight.h>


// hkxLight ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxLight)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(LightType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxLight)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxLight, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkxLight, LightType, s_libraryName)

#include <Common/SceneData/Material/hkxMaterial.h>


// hkxMaterial ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxMaterial)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(TextureStage)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Property)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(TextureType)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(PropertyKey)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(UVMappingAlgorithm)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Transparency)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxMaterial)
    HK_TRACKER_MEMBER(hkxMaterial, m_name, 0, "hkStringPtr") // hkStringPtr
    HK_TRACKER_MEMBER(hkxMaterial, m_stages, 0, "hkArray<hkxMaterial::TextureStage, hkContainerHeapAllocator>") // hkArray< struct hkxMaterial::TextureStage, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkxMaterial, m_subMaterials, 0, "hkArray<hkxMaterial *, hkContainerHeapAllocator>") // hkArray< class hkxMaterial *, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkxMaterial, m_extraData, 0, "hkReferencedObject *") // class hkReferencedObject *
    HK_TRACKER_MEMBER(hkxMaterial, m_properties, 0, "hkArray<hkxMaterial::Property, hkContainerHeapAllocator>") // hkArray< struct hkxMaterial::Property, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxMaterial, s_libraryName, hkxAttributeHolder)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkxMaterial, TextureType, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkxMaterial, PropertyKey, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkxMaterial, UVMappingAlgorithm, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkxMaterial, Transparency, s_libraryName)


// TextureStage hkxMaterial

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxMaterial::TextureStage)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxMaterial::TextureStage)
    HK_TRACKER_MEMBER(hkxMaterial::TextureStage, m_texture, 0, "hkReferencedObject *") // class hkReferencedObject *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkxMaterial::TextureStage, s_libraryName)


// Property hkxMaterial
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkxMaterial, Property, s_libraryName)

#include <Common/SceneData/Material/hkxMaterialEffect.h>


// hkxMaterialEffect ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxMaterialEffect)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(EffectType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxMaterialEffect)
    HK_TRACKER_MEMBER(hkxMaterialEffect, m_name, 0, "hkStringPtr") // hkStringPtr
    HK_TRACKER_MEMBER(hkxMaterialEffect, m_data, 0, "hkArray<hkUint8, hkContainerHeapAllocator>") // hkArray< hkUint8, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxMaterialEffect, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkxMaterialEffect, EffectType, s_libraryName)

#include <Common/SceneData/Material/hkxMaterialShader.h>


// hkxMaterialShader ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxMaterialShader)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ShaderType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxMaterialShader)
    HK_TRACKER_MEMBER(hkxMaterialShader, m_name, 0, "hkStringPtr") // hkStringPtr
    HK_TRACKER_MEMBER(hkxMaterialShader, m_vertexEntryName, 0, "hkStringPtr") // hkStringPtr
    HK_TRACKER_MEMBER(hkxMaterialShader, m_geomEntryName, 0, "hkStringPtr") // hkStringPtr
    HK_TRACKER_MEMBER(hkxMaterialShader, m_pixelEntryName, 0, "hkStringPtr") // hkStringPtr
    HK_TRACKER_MEMBER(hkxMaterialShader, m_data, 0, "hkArray<hkUint8, hkContainerHeapAllocator>") // hkArray< hkUint8, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxMaterialShader, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkxMaterialShader, ShaderType, s_libraryName)

#include <Common/SceneData/Material/hkxMaterialShaderSet.h>


// hkxMaterialShaderSet ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxMaterialShaderSet)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxMaterialShaderSet)
    HK_TRACKER_MEMBER(hkxMaterialShaderSet, m_shaders, 0, "hkArray<hkxMaterialShader *, hkContainerHeapAllocator>") // hkArray< class hkxMaterialShader *, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxMaterialShaderSet, s_libraryName, hkReferencedObject)

#include <Common/SceneData/Material/hkxTextureFile.h>


// hkxTextureFile ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxTextureFile)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxTextureFile)
    HK_TRACKER_MEMBER(hkxTextureFile, m_filename, 0, "hkStringPtr") // hkStringPtr
    HK_TRACKER_MEMBER(hkxTextureFile, m_name, 0, "hkStringPtr") // hkStringPtr
    HK_TRACKER_MEMBER(hkxTextureFile, m_originalFilename, 0, "hkStringPtr") // hkStringPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxTextureFile, s_libraryName, hkReferencedObject)

#include <Common/SceneData/Material/hkxTextureInplace.h>


// hkxTextureInplace ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxTextureInplace)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxTextureInplace)
    HK_TRACKER_MEMBER(hkxTextureInplace, m_data, 0, "hkArray<hkUint8, hkContainerHeapAllocator>") // hkArray< hkUint8, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkxTextureInplace, m_name, 0, "hkStringPtr") // hkStringPtr
    HK_TRACKER_MEMBER(hkxTextureInplace, m_originalFilename, 0, "hkStringPtr") // hkStringPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxTextureInplace, s_libraryName, hkReferencedObject)

#include <Common/SceneData/Mesh/Channels/hkxEdgeSelectionChannel.h>


// hkxEdgeSelectionChannel ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxEdgeSelectionChannel)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxEdgeSelectionChannel)
    HK_TRACKER_MEMBER(hkxEdgeSelectionChannel, m_selectedEdges, 0, "hkArray<hkInt32, hkContainerHeapAllocator>") // hkArray< hkInt32, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxEdgeSelectionChannel, s_libraryName, hkReferencedObject)

#include <Common/SceneData/Mesh/Channels/hkxTriangleSelectionChannel.h>


// hkxTriangleSelectionChannel ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxTriangleSelectionChannel)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxTriangleSelectionChannel)
    HK_TRACKER_MEMBER(hkxTriangleSelectionChannel, m_selectedTriangles, 0, "hkArray<hkInt32, hkContainerHeapAllocator>") // hkArray< hkInt32, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxTriangleSelectionChannel, s_libraryName, hkReferencedObject)

#include <Common/SceneData/Mesh/Channels/hkxVertexFloatDataChannel.h>


// hkxVertexFloatDataChannel ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxVertexFloatDataChannel)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(VertexFloatDimensions)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxVertexFloatDataChannel)
    HK_TRACKER_MEMBER(hkxVertexFloatDataChannel, m_perVertexFloats, 0, "hkArray<float, hkContainerHeapAllocator>") // hkArray< float, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxVertexFloatDataChannel, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkxVertexFloatDataChannel, VertexFloatDimensions, s_libraryName)

#include <Common/SceneData/Mesh/Channels/hkxVertexIntDataChannel.h>


// hkxVertexIntDataChannel ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxVertexIntDataChannel)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxVertexIntDataChannel)
    HK_TRACKER_MEMBER(hkxVertexIntDataChannel, m_perVertexInts, 0, "hkArray<hkInt32, hkContainerHeapAllocator>") // hkArray< hkInt32, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxVertexIntDataChannel, s_libraryName, hkReferencedObject)

#include <Common/SceneData/Mesh/Channels/hkxVertexSelectionChannel.h>


// hkxVertexSelectionChannel ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxVertexSelectionChannel)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxVertexSelectionChannel)
    HK_TRACKER_MEMBER(hkxVertexSelectionChannel, m_selectedVertices, 0, "hkArray<hkInt32, hkContainerHeapAllocator>") // hkArray< hkInt32, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxVertexSelectionChannel, s_libraryName, hkReferencedObject)

#include <Common/SceneData/Mesh/Channels/hkxVertexVectorDataChannel.h>


// hkxVertexVectorDataChannel ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxVertexVectorDataChannel)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxVertexVectorDataChannel)
    HK_TRACKER_MEMBER(hkxVertexVectorDataChannel, m_perVertexVectors, 0, "hkArray<float, hkContainerHeapAllocator>") // hkArray< float, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxVertexVectorDataChannel, s_libraryName, hkReferencedObject)

#include <Common/SceneData/Mesh/MemoryMeshFactory/hkxMemoryMeshFactory.h>


// hkxMemoryMeshFactory ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxMemoryMeshFactory)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxMemoryMeshFactory)
    HK_TRACKER_MEMBER(hkxMemoryMeshFactory, m_meshSystem, 0, "hkMemoryMeshSystem*") // class hkMemoryMeshSystem*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxMemoryMeshFactory, s_libraryName, hkxMeshFactory)

#include <Common/SceneData/Mesh/hkxIndexBuffer.h>


// hkxIndexBuffer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxIndexBuffer)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(IndexType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxIndexBuffer)
    HK_TRACKER_MEMBER(hkxIndexBuffer, m_indices16, 0, "hkArray<hkUint16, hkContainerHeapAllocator>") // hkArray< hkUint16, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkxIndexBuffer, m_indices32, 0, "hkArray<hkUint32, hkContainerHeapAllocator>") // hkArray< hkUint32, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxIndexBuffer, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkxIndexBuffer, IndexType, s_libraryName)

#include <Common/SceneData/Mesh/hkxMesh.h>


// hkxMesh ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxMesh)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(UserChannelInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxMesh)
    HK_TRACKER_MEMBER(hkxMesh, m_sections, 0, "hkArray<hkxMeshSection *, hkContainerHeapAllocator>") // hkArray< class hkxMeshSection *, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkxMesh, m_userChannelInfos, 0, "hkArray<hkxMesh::UserChannelInfo *, hkContainerHeapAllocator>") // hkArray< class hkxMesh::UserChannelInfo *, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxMesh, s_libraryName, hkReferencedObject)


// UserChannelInfo hkxMesh

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxMesh::UserChannelInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxMesh::UserChannelInfo)
    HK_TRACKER_MEMBER(hkxMesh::UserChannelInfo, m_name, 0, "hkStringPtr") // hkStringPtr
    HK_TRACKER_MEMBER(hkxMesh::UserChannelInfo, m_className, 0, "hkStringPtr") // hkStringPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxMesh::UserChannelInfo, s_libraryName, hkxAttributeHolder)

#include <Common/SceneData/Mesh/hkxMeshFactory.h>


// hkxMeshFactory ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxMeshFactory)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxMeshFactory)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkxMeshFactory, s_libraryName, hkReferencedObject)

#include <Common/SceneData/Mesh/hkxMeshSection.h>


// hkxMeshSection ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxMeshSection)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxMeshSection)
    HK_TRACKER_MEMBER(hkxMeshSection, m_vertexBuffer, 0, "hkxVertexBuffer *") // class hkxVertexBuffer *
    HK_TRACKER_MEMBER(hkxMeshSection, m_indexBuffers, 0, "hkArray<hkxIndexBuffer *, hkContainerHeapAllocator>") // hkArray< class hkxIndexBuffer *, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkxMeshSection, m_material, 0, "hkxMaterial *") // class hkxMaterial *
    HK_TRACKER_MEMBER(hkxMeshSection, m_userChannels, 0, "hkArray<hkReferencedObject *, hkContainerHeapAllocator>") // hkArray< class hkReferencedObject *, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkxMeshSection, m_vertexAnimations, 0, "hkArray<hkxVertexAnimation *, hkContainerHeapAllocator>") // hkArray< class hkxVertexAnimation *, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkxMeshSection, m_linearKeyFrameHints, 0, "hkArray<float, hkContainerHeapAllocator>") // hkArray< float, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxMeshSection, s_libraryName, hkReferencedObject)

#include <Common/SceneData/Mesh/hkxMeshSectionUtil.h>


// hkxBoneIndicesInt8Data ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxBoneIndicesInt8Data)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BoneIndicesDataPtr)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxBoneIndicesInt8Data)
    HK_TRACKER_MEMBER(hkxBoneIndicesInt8Data, m_data, 0, "hkxBoneIndicesInt8Data::BoneIndicesDataPtr") // struct hkxBoneIndicesInt8Data::BoneIndicesDataPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkxBoneIndicesInt8Data, s_libraryName)


// BoneIndicesDataPtr hkxBoneIndicesInt8Data

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxBoneIndicesInt8Data::BoneIndicesDataPtr)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxBoneIndicesInt8Data::BoneIndicesDataPtr)
    HK_TRACKER_MEMBER(hkxBoneIndicesInt8Data::BoneIndicesDataPtr, m_boneIndicesPtr, 0, "hkUint8*") // hkUint8*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkxBoneIndicesInt8Data::BoneIndicesDataPtr, s_libraryName)


// hkxBoneIndicesInt16Data ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxBoneIndicesInt16Data)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BoneIndicesDataPtr)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxBoneIndicesInt16Data)
    HK_TRACKER_MEMBER(hkxBoneIndicesInt16Data, m_data, 0, "hkxBoneIndicesInt16Data::BoneIndicesDataPtr") // struct hkxBoneIndicesInt16Data::BoneIndicesDataPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkxBoneIndicesInt16Data, s_libraryName)


// BoneIndicesDataPtr hkxBoneIndicesInt16Data

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxBoneIndicesInt16Data::BoneIndicesDataPtr)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxBoneIndicesInt16Data::BoneIndicesDataPtr)
    HK_TRACKER_MEMBER(hkxBoneIndicesInt16Data::BoneIndicesDataPtr, m_boneIndicesPtr, 0, "hkUint16*") // hkUint16*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkxBoneIndicesInt16Data::BoneIndicesDataPtr, s_libraryName)


// hkxMeshSectionUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkxMeshSectionUtil, s_libraryName)

#include <Common/SceneData/Mesh/hkxVertexAnimation.h>


// hkxVertexAnimation ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxVertexAnimation)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(UsageMap)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxVertexAnimation)
    HK_TRACKER_MEMBER(hkxVertexAnimation, m_vertData, 0, "hkxVertexBuffer") // class hkxVertexBuffer
    HK_TRACKER_MEMBER(hkxVertexAnimation, m_vertexIndexMap, 0, "hkArray<hkInt32, hkContainerHeapAllocator>") // hkArray< hkInt32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkxVertexAnimation, m_componentMap, 0, "hkArray<hkxVertexAnimation::UsageMap, hkContainerHeapAllocator>") // hkArray< struct hkxVertexAnimation::UsageMap, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxVertexAnimation, s_libraryName, hkReferencedObject)


// UsageMap hkxVertexAnimation
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkxVertexAnimation, UsageMap, s_libraryName)


// hkxVertexAnimationStateCache ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxVertexAnimationStateCache)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxVertexAnimationStateCache)
    HK_TRACKER_MEMBER(hkxVertexAnimationStateCache, m_state, 0, "hkxVertexBuffer*") // class hkxVertexBuffer*
    HK_TRACKER_MEMBER(hkxVertexAnimationStateCache, m_alteredVerts, 0, "hkBitField") // class hkBitField
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxVertexAnimationStateCache, s_libraryName, hkReferencedObject)

#include <Common/SceneData/Mesh/hkxVertexBuffer.h>


// hkxVertexBuffer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxVertexBuffer)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(VertexData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxVertexBuffer)
    HK_TRACKER_MEMBER(hkxVertexBuffer, m_data, 0, "hkxVertexBuffer::VertexData") // struct hkxVertexBuffer::VertexData
    HK_TRACKER_MEMBER(hkxVertexBuffer, m_desc, 0, "hkxVertexDescription") // class hkxVertexDescription
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxVertexBuffer, s_libraryName, hkReferencedObject)


// VertexData hkxVertexBuffer

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxVertexBuffer::VertexData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxVertexBuffer::VertexData)
    HK_TRACKER_MEMBER(hkxVertexBuffer::VertexData, m_vectorData, 0, "hkArray<float, hkContainerHeapAllocator>") // hkArray< float, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkxVertexBuffer::VertexData, m_floatData, 0, "hkArray<float, hkContainerHeapAllocator>") // hkArray< float, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkxVertexBuffer::VertexData, m_uint32Data, 0, "hkArray<hkUint32, hkContainerHeapAllocator>") // hkArray< hkUint32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkxVertexBuffer::VertexData, m_uint16Data, 0, "hkArray<hkUint16, hkContainerHeapAllocator>") // hkArray< hkUint16, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkxVertexBuffer::VertexData, m_uint8Data, 0, "hkArray<hkUint8, hkContainerHeapAllocator>") // hkArray< hkUint8, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkxVertexBuffer::VertexData, s_libraryName)

#include <Common/SceneData/Mesh/hkxVertexDescription.h>


// hkxVertexDescription ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxVertexDescription)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ElementDecl)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(DataType)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(DataUsage)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(DataHint)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxVertexDescription)
    HK_TRACKER_MEMBER(hkxVertexDescription, m_decls, 0, "hkArray<hkxVertexDescription::ElementDecl, hkContainerHeapAllocator>") // hkArray< struct hkxVertexDescription::ElementDecl, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkxVertexDescription, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkxVertexDescription, DataType, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkxVertexDescription, DataUsage, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkxVertexDescription, DataHint, s_libraryName)


// ElementDecl hkxVertexDescription
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkxVertexDescription, ElementDecl, s_libraryName)

#include <Common/SceneData/Mesh/hkxVertexUtil.h>


// hkxVertexUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkxVertexUtil, s_libraryName)

#include <Common/SceneData/Scene/hkxScene.h>


// hkxScene ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxScene)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxScene)
    HK_TRACKER_MEMBER(hkxScene, m_modeller, 0, "hkStringPtr") // hkStringPtr
    HK_TRACKER_MEMBER(hkxScene, m_asset, 0, "hkStringPtr") // hkStringPtr
    HK_TRACKER_MEMBER(hkxScene, m_rootNode, 0, "hkxNode *") // class hkxNode *
    HK_TRACKER_MEMBER(hkxScene, m_selectionSets, 0, "hkArray<hkxNodeSelectionSet *, hkContainerHeapAllocator>") // hkArray< class hkxNodeSelectionSet *, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkxScene, m_cameras, 0, "hkArray<hkxCamera *, hkContainerHeapAllocator>") // hkArray< class hkxCamera *, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkxScene, m_lights, 0, "hkArray<hkxLight *, hkContainerHeapAllocator>") // hkArray< class hkxLight *, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkxScene, m_meshes, 0, "hkArray<hkxMesh *, hkContainerHeapAllocator>") // hkArray< class hkxMesh *, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkxScene, m_materials, 0, "hkArray<hkxMaterial *, hkContainerHeapAllocator>") // hkArray< class hkxMaterial *, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkxScene, m_inplaceTextures, 0, "hkArray<hkxTextureInplace *, hkContainerHeapAllocator>") // hkArray< class hkxTextureInplace *, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkxScene, m_externalTextures, 0, "hkArray<hkxTextureFile *, hkContainerHeapAllocator>") // hkArray< class hkxTextureFile *, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkxScene, m_skinBindings, 0, "hkArray<hkxSkinBinding *, hkContainerHeapAllocator>") // hkArray< class hkxSkinBinding *, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkxScene, m_splines, 0, "hkArray<hkxSpline *, hkContainerHeapAllocator>") // hkArray< class hkxSpline *, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxScene, s_libraryName, hkReferencedObject)

#include <Common/SceneData/Scene/hkxSceneUtils.h>


// hkxSceneUtils ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxSceneUtils)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SceneTransformOptions)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(GraphicsNode)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(TransformInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkxSceneUtils, s_libraryName)


// SceneTransformOptions hkxSceneUtils
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkxSceneUtils, SceneTransformOptions, s_libraryName)


// GraphicsNode hkxSceneUtils

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxSceneUtils::GraphicsNode)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxSceneUtils::GraphicsNode)
    HK_TRACKER_MEMBER(hkxSceneUtils::GraphicsNode, m_node, 0, "hkxNode*") // class hkxNode*
    HK_TRACKER_MEMBER(hkxSceneUtils::GraphicsNode, m_name, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkxSceneUtils::GraphicsNode, s_libraryName)


// TransformInfo hkxSceneUtils
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkxSceneUtils, TransformInfo, s_libraryName)

#include <Common/SceneData/SceneDataToGeometryConverter/hkxSceneDataToGeometryConverter.h>


// hkxSceneDataToGeometryConverter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxSceneDataToGeometryConverter)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(GeometryInstances)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkxSceneDataToGeometryConverter, s_libraryName)


// GeometryInstances hkxSceneDataToGeometryConverter

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxSceneDataToGeometryConverter::GeometryInstances)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Instance)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxSceneDataToGeometryConverter::GeometryInstances)
    HK_TRACKER_MEMBER(hkxSceneDataToGeometryConverter::GeometryInstances, m_instances, 0, "hkArray<hkxSceneDataToGeometryConverter::GeometryInstances::Instance, hkContainerHeapAllocator>") // hkArray< struct hkxSceneDataToGeometryConverter::GeometryInstances::Instance, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkxSceneDataToGeometryConverter::GeometryInstances, m_geometries, 0, "hkArray<hkGeometry, hkContainerHeapAllocator>") // hkArray< struct hkGeometry, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkxSceneDataToGeometryConverter::GeometryInstances, s_libraryName)


// Instance hkxSceneDataToGeometryConverter::GeometryInstances
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkxSceneDataToGeometryConverter::GeometryInstances, Instance, s_libraryName)

#include <Common/SceneData/Selection/hkxNodeSelectionSet.h>


// hkxNodeSelectionSet ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxNodeSelectionSet)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxNodeSelectionSet)
    HK_TRACKER_MEMBER(hkxNodeSelectionSet, m_selectedNodes, 0, "hkArray<hkxNode *, hkContainerHeapAllocator>") // hkArray< class hkxNode *, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkxNodeSelectionSet, m_name, 0, "hkStringPtr") // hkStringPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxNodeSelectionSet, s_libraryName, hkxAttributeHolder)

#include <Common/SceneData/Skin/hkxSkinBinding.h>


// hkxSkinBinding ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxSkinBinding)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxSkinBinding)
    HK_TRACKER_MEMBER(hkxSkinBinding, m_mesh, 0, "hkxMesh *") // class hkxMesh *
    HK_TRACKER_MEMBER(hkxSkinBinding, m_nodeNames, 0, "hkArray<hkStringPtr, hkContainerHeapAllocator>") // hkArray< hkStringPtr, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkxSkinBinding, m_bindPose, 0, "hkArray<hkMatrix4f, hkContainerHeapAllocator>") // hkArray< hkMatrix4f, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxSkinBinding, s_libraryName, hkReferencedObject)

#include <Common/SceneData/Skin/hkxSkinUtils.h>


// hkxSkinUtils ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkxSkinUtils, s_libraryName)

#include <Common/SceneData/Spline/hkxSpline.h>


// hkxSpline ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxSpline)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ControlPoint)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ControlType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxSpline)
    HK_TRACKER_MEMBER(hkxSpline, m_controlPoints, 0, "hkArray<hkxSpline::ControlPoint, hkContainerHeapAllocator>") // hkArray< struct hkxSpline::ControlPoint, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxSpline, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkxSpline, ControlType, s_libraryName)


// ControlPoint hkxSpline
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkxSpline, ControlPoint, s_libraryName)

#include <Common/SceneData/VisualDebugger/Viewer/hkxSceneViewer.h>


// hkxSceneViewer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxSceneViewer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxSceneViewer)
    HK_TRACKER_MEMBER(hkxSceneViewer, m_context, 0, "hkxSceneDataContext*") // class hkxSceneDataContext*
    HK_TRACKER_MEMBER(hkxSceneViewer, m_displayGeometryBuilder, 0, "hkDisplayGeometryBuilder*") // class hkDisplayGeometryBuilder*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxSceneViewer, s_libraryName, hkReferencedObject)

#include <Common/SceneData/VisualDebugger/hkxSceneDataContext.h>


// hkxSceneDataContextListener ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxSceneDataContextListener)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxSceneDataContextListener)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkxSceneDataContextListener, s_libraryName)


// hkxSceneDataContext ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkxSceneDataContext)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkxSceneDataContext)
    HK_TRACKER_MEMBER(hkxSceneDataContext, m_scenes, 0, "hkArray<hkxScene*, hkContainerHeapAllocator>") // hkArray< class hkxScene*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkxSceneDataContext, m_listeners, 0, "hkArray<hkxSceneDataContextListener*, hkContainerHeapAllocator>") // hkArray< class hkxSceneDataContextListener*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkxSceneDataContext, m_searchPaths, 0, "hkArray<char*, hkContainerHeapAllocator>") // hkArray< const char*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkxSceneDataContext, s_libraryName, hkReferencedObject)

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
