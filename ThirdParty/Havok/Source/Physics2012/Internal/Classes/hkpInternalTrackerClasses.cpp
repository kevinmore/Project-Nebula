/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#include <Physics2012/Internal/hkpInternal.h>
static const char s_libraryName[] = "hkpInternal";
#include <Common/Base/Memory/Tracker/hkTrackerClassDefinition.h>

void HK_CALL hkpInternalRegister() {}

#include <Physics2012/Internal/Collide/BvCompressedMesh/hkpBvCompressedMeshShape.h>


// hkpBvCompressedMeshShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBvCompressedMeshShape)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(PerPrimitiveDataMode)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(PrimitiveType)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Config)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBvCompressedMeshShape)
    HK_TRACKER_MEMBER(hkpBvCompressedMeshShape, m_collisionFilterInfoPalette, 0, "hkArray<hkUint32, hkContainerHeapAllocator>") // hkArray< hkUint32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpBvCompressedMeshShape, m_userDataPalette, 0, "hkArray<hkUint32, hkContainerHeapAllocator>") // hkArray< hkUint32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpBvCompressedMeshShape, m_userStringPalette, 0, "hkArray<hkStringPtr, hkContainerHeapAllocator>") // hkArray< hkStringPtr, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpBvCompressedMeshShape, s_libraryName, hkpBvTreeShape)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpBvCompressedMeshShape, PerPrimitiveDataMode, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpBvCompressedMeshShape, PrimitiveType, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpBvCompressedMeshShape, Config, s_libraryName)

#include <Physics2012/Internal/Collide/BvCompressedMesh/hkpBvCompressedMeshShapeCinfo.h>


// hkpBvCompressedMeshShapeCinfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBvCompressedMeshShapeCinfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBvCompressedMeshShapeCinfo)
    HK_TRACKER_MEMBER(hkpBvCompressedMeshShapeCinfo, m_triangleIndexToShapeKeyMap, 0, "hkArray<hkUint32, hkContainerHeapAllocator>*") // hkArray< hkUint32, struct hkContainerHeapAllocator >*
    HK_TRACKER_MEMBER(hkpBvCompressedMeshShapeCinfo, m_convexShapeIndexToShapeKeyMap, 0, "hkArray<hkUint32, hkContainerHeapAllocator>*") // hkArray< hkUint32, struct hkContainerHeapAllocator >*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkpBvCompressedMeshShapeCinfo, s_libraryName)


// hkpDefaultBvCompressedMeshShapeCinfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpDefaultBvCompressedMeshShapeCinfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ConvexShapeInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpDefaultBvCompressedMeshShapeCinfo)
    HK_TRACKER_MEMBER(hkpDefaultBvCompressedMeshShapeCinfo, m_geometry, 0, "hkGeometry*") // const struct hkGeometry*
    HK_TRACKER_MEMBER(hkpDefaultBvCompressedMeshShapeCinfo, m_shapes, 0, "hkArray<hkpDefaultBvCompressedMeshShapeCinfo::ConvexShapeInfo, hkContainerHeapAllocator>") // hkArray< struct hkpDefaultBvCompressedMeshShapeCinfo::ConvexShapeInfo, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpDefaultBvCompressedMeshShapeCinfo, s_libraryName, hkpBvCompressedMeshShapeCinfo)


// ConvexShapeInfo hkpDefaultBvCompressedMeshShapeCinfo

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpDefaultBvCompressedMeshShapeCinfo::ConvexShapeInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpDefaultBvCompressedMeshShapeCinfo::ConvexShapeInfo)
    HK_TRACKER_MEMBER(hkpDefaultBvCompressedMeshShapeCinfo::ConvexShapeInfo, m_shape, 0, "hkpConvexShape*") // const class hkpConvexShape*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpDefaultBvCompressedMeshShapeCinfo::ConvexShapeInfo, s_libraryName)

#include <Physics2012/Internal/Collide/ConvexPieceMesh/hkpConvexPieceMeshBuilder.h>


// hkpShapeCollectionMaterialMediator ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpShapeCollectionMaterialMediator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpShapeCollectionMaterialMediator)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkpShapeCollectionMaterialMediator, s_libraryName)


// hkpDefaultShapeCollectionMaterialMediator ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpDefaultShapeCollectionMaterialMediator)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpDefaultShapeCollectionMaterialMediator)
    HK_TRACKER_MEMBER(hkpDefaultShapeCollectionMaterialMediator, m_meshShape, 0, "hkpMeshShape*") // class hkpMeshShape*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpDefaultShapeCollectionMaterialMediator, s_libraryName, hkpShapeCollectionMaterialMediator)


// hkpConvexPieceMeshBuilder ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConvexPieceMeshBuilder)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(TriangleInfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BuilderInput)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ConvexPiece)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConvexPieceMeshBuilder)
    HK_TRACKER_MEMBER(hkpConvexPieceMeshBuilder, m_convexPieces, 0, "hkArray<hkpConvexPieceMeshBuilder::ConvexPiece*, hkContainerHeapAllocator>") // hkArray< struct hkpConvexPieceMeshBuilder::ConvexPiece*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpConvexPieceMeshBuilder, s_libraryName, hkReferencedObject)


// TriangleInfo hkpConvexPieceMeshBuilder
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpConvexPieceMeshBuilder, TriangleInfo, s_libraryName)


// BuilderInput hkpConvexPieceMeshBuilder

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConvexPieceMeshBuilder::BuilderInput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConvexPieceMeshBuilder::BuilderInput)
    HK_TRACKER_MEMBER(hkpConvexPieceMeshBuilder::BuilderInput, m_moppCollidable, 0, "hkpCollidable*") // class hkpCollidable*
    HK_TRACKER_MEMBER(hkpConvexPieceMeshBuilder::BuilderInput, m_sphereCollidable, 0, "hkpCollidable*") // class hkpCollidable*
    HK_TRACKER_MEMBER(hkpConvexPieceMeshBuilder::BuilderInput, m_sphereTransform, 0, "hkTransform*") // hkTransform*
    HK_TRACKER_MEMBER(hkpConvexPieceMeshBuilder::BuilderInput, m_meshShape, 0, "hkpShapeCollection*") // const class hkpShapeCollection*
    HK_TRACKER_MEMBER(hkpConvexPieceMeshBuilder::BuilderInput, m_materialMediator, 0, "hkpShapeCollectionMaterialMediator*") // const class hkpShapeCollectionMaterialMediator*
    HK_TRACKER_MEMBER(hkpConvexPieceMeshBuilder::BuilderInput, m_shapeKeyToTriangleInfoIndex, 0, "hkPointerMap<hkUint32, hkUint32, hkContainerHeapAllocator>") // class hkPointerMap< hkUint32, hkUint32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpConvexPieceMeshBuilder::BuilderInput, m_triangleInfo, 0, "hkArray<hkpConvexPieceMeshBuilder::TriangleInfo, hkContainerHeapAllocator>") // hkArray< struct hkpConvexPieceMeshBuilder::TriangleInfo, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpConvexPieceMeshBuilder::BuilderInput, m_collisionInput, 0, "hkpLinearCastCollisionInput") // struct hkpLinearCastCollisionInput
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpConvexPieceMeshBuilder::BuilderInput, s_libraryName)


// ConvexPiece hkpConvexPieceMeshBuilder

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConvexPieceMeshBuilder::ConvexPiece)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConvexPieceMeshBuilder::ConvexPiece)
    HK_TRACKER_MEMBER(hkpConvexPieceMeshBuilder::ConvexPiece, m_triangles, 0, "hkArray<hkUint32, hkContainerHeapAllocator>") // hkArray< hkUint32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpConvexPieceMeshBuilder::ConvexPiece, m_vertices, 0, "hkArray<hkVector4f, hkContainerHeapAllocator>") // hkArray< hkVector4f, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpConvexPieceMeshBuilder::ConvexPiece, s_libraryName)

#include <Physics2012/Internal/Collide/ConvexPieceMesh/hkpConvexPieceStreamData.h>


// hkpConvexPieceStreamData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConvexPieceStreamData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConvexPieceStreamData)
    HK_TRACKER_MEMBER(hkpConvexPieceStreamData, m_convexPieceStream, 0, "hkArray<hkUint32, hkContainerHeapAllocator>") // hkArray< hkUint32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpConvexPieceStreamData, m_convexPieceOffsets, 0, "hkArray<hkUint32, hkContainerHeapAllocator>") // hkArray< hkUint32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpConvexPieceStreamData, m_convexPieceSingleTriangles, 0, "hkArray<hkUint32, hkContainerHeapAllocator>") // hkArray< hkUint32, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpConvexPieceStreamData, s_libraryName, hkReferencedObject)

#include <Physics2012/Internal/Collide/Gjk/GskManifold/hkpGskManifold.h>


// hkpGskManifold ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpGskManifold)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ContactPoint)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkpGskManifold, s_libraryName)


// ContactPoint hkpGskManifold
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpGskManifold, ContactPoint, s_libraryName)

#include <Physics2012/Internal/Collide/Gjk/hkpGjkCache.h>


// hkpGjkCache ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpGjkCache, s_libraryName)

#include <Physics2012/Internal/Collide/Gjk/hkpGskCache.h>


// hkpGskCache ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpGskCache, s_libraryName)


// hkGskCache16 ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkGskCache16, s_libraryName)

#include <Physics2012/Internal/Collide/Mopp/Code/hkpMoppCode.h>


// hkpMoppChunk ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpMoppChunk, s_libraryName)


// hkpMoppCodeReindexedTerminal ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpMoppCodeReindexedTerminal, s_libraryName)


// hkpMoppCode ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMoppCode)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CodeInfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BuildType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMoppCode)
    HK_TRACKER_MEMBER(hkpMoppCode, m_data, 0, "hkArray<hkUint8, hkContainerHeapAllocator>") // hkArray< hkUint8, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpMoppCode, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpMoppCode, BuildType, s_libraryName)


// CodeInfo hkpMoppCode
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpMoppCode, CodeInfo, s_libraryName)

#include <Physics2012/Internal/Collide/Mopp/Machine/hkp26Dop.h>


// hkp26Dop ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkp26Dop, s_libraryName)

#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppMachine.h>


// hkpMoppPlanesQueryInput ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMoppPlanesQueryInput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMoppPlanesQueryInput)
    HK_TRACKER_MEMBER(hkpMoppPlanesQueryInput, m_planes, 0, "hkVector4*") // const hkVector4*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpMoppPlanesQueryInput, s_libraryName)


// hkpMoppInfo ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpMoppInfo, s_libraryName)


// hkpMoppKDopQuery ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpMoppKDopQuery, s_libraryName)

#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppModifier.h>

// hk.MemoryTracker ignore hkpMoppModifier
#include <Physics2012/Internal/Collide/StaticCompound/hkpShapeKeyTable.h>


// hkpShapeKeyTable ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpShapeKeyTable)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Block)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpShapeKeyTable)
    HK_TRACKER_MEMBER(hkpShapeKeyTable, m_lists, 0, "hkpShapeKeyTable::Block*") // struct hkpShapeKeyTable::Block*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpShapeKeyTable, s_libraryName)


// Block hkpShapeKeyTable

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpShapeKeyTable::Block)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpShapeKeyTable::Block)
    HK_TRACKER_MEMBER(hkpShapeKeyTable::Block, m_next, 0, "hkpShapeKeyTable::Block*") // struct hkpShapeKeyTable::Block*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpShapeKeyTable::Block, s_libraryName)

#include <Physics2012/Internal/Collide/StaticCompound/hkpStaticCompoundShape.h>


// hkpStaticCompoundShape ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpStaticCompoundShape)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Instance)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Config)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpStaticCompoundShape)
    HK_TRACKER_MEMBER(hkpStaticCompoundShape, m_instances, 0, "hkArray<hkpStaticCompoundShape::Instance, hkContainerHeapAllocator>") // hkArray< struct hkpStaticCompoundShape::Instance, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpStaticCompoundShape, m_instanceExtraInfos, 0, "hkArray<hkUint16, hkContainerHeapAllocator>") // hkArray< hkUint16, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpStaticCompoundShape, m_disabledLargeShapeKeyTable, 0, "hkpShapeKeyTable") // class hkpShapeKeyTable
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpStaticCompoundShape, s_libraryName, hkpBvTreeShape)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpStaticCompoundShape, Config, s_libraryName)


// Instance hkpStaticCompoundShape

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpStaticCompoundShape::Instance)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpStaticCompoundShape::Instance)
    HK_TRACKER_MEMBER(hkpStaticCompoundShape::Instance, m_shape, 0, "hkpShape*") // const class hkpShape*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpStaticCompoundShape::Instance, s_libraryName)

#include <Physics2012/Internal/Dynamics/Constraints/hkpConstraintProjector.h>


// hkpConstraintProjector ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpConstraintProjector, s_libraryName)

#include <Physics2012/Internal/Solver/Contact/hkpSimpleContactConstraintInfo.h>


// hkpSimpleContactConstraintDataInfo ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpSimpleContactConstraintDataInfo, s_libraryName)


// hkpSimpleContactConstraintDataInfoInternal ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpSimpleContactConstraintDataInfoInternal, s_libraryName)

#include <Physics2012/Internal/Solver/SimpleConstraints/hkpSimpleConstraintUtil.h>


// hkpSimpleConstraintInfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSimpleConstraintInfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BodyInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSimpleConstraintInfo)
    HK_TRACKER_MEMBER(hkpSimpleConstraintInfo, m_bodyInfo, 0, "hkpSimpleConstraintInfo::BodyInfo [2]") // struct hkpSimpleConstraintInfo::BodyInfo [2]
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpSimpleConstraintInfo, s_libraryName)


// BodyInfo hkpSimpleConstraintInfo

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSimpleConstraintInfo::BodyInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSimpleConstraintInfo::BodyInfo)
    HK_TRACKER_MEMBER(hkpSimpleConstraintInfo::BodyInfo, m_transform, 0, "hkTransform*") // const hkTransform*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpSimpleConstraintInfo::BodyInfo, s_libraryName)


// hkpSimpleConstraintInfoInitInput ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSimpleConstraintInfoInitInput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSimpleConstraintInfoInitInput)
    HK_TRACKER_MEMBER(hkpSimpleConstraintInfoInitInput, m_transform, 0, "hkTransform*") // const hkTransform*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpSimpleConstraintInfoInitInput, s_libraryName)


// hkpSimpleConstraintUtilCollideParams ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpSimpleConstraintUtilCollideParams, s_libraryName)


// hkpBodyVelocity ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpBodyVelocity, s_libraryName)

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
