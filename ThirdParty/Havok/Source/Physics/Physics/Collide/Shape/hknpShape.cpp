/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Physics/Physics/Collide/Shape/hknpShape.h>
#include <Physics/Physics/Collide/Shape/hknpShapeUtil.h>
#include <Physics/Physics/Collide/Shape/hknpShapeCollector.h>
#include <Common/GeometryUtilities/Inertia/hkInertiaTensorComputer.h>
#include <Physics/Physics/Dynamics/Simulation/hknpSimulationThreadContext.h>


#if !defined(HK_PLATFORM_SPU)
#	define HKNP_SHAPE_NOT_IMPLEMENTED(ID)	HK_ERROR(ID, "Not implemented");
#else
//	Not issuing an error on SPU saves us some code (~1KB in total)
#	define HKNP_SHAPE_NOT_IMPLEMENTED(ID)		HK_BREAKPOINT(ID);
#endif

hknpShapeType::Enum hknpShape::getType() const
{
	HKNP_SHAPE_NOT_IMPLEMENTED(0x4cc169fa);
	return hknpShapeType::INVALID;
}

void hknpShape::calcAabb( const hkTransform& transform, hkAabb& aabbOut ) const
{
	HKNP_SHAPE_NOT_IMPLEMENTED(0x7fe8e736);
}

int hknpShape::calcSize() const
{
	HKNP_SHAPE_NOT_IMPLEMENTED(0x2667021d);
	return 0;
}

#if !defined( HK_PLATFORM_SPU )

hkRefNew<hknpShapeKeyIterator> hknpShape::createShapeKeyIterator( const hknpShapeKeyMask* mask ) const
{
	HKNP_SHAPE_NOT_IMPLEMENTED(0x57286ee2);
	return HK_NULL;
}

#else

hknpShapeKeyIterator* hknpShape::createShapeKeyIterator( hkUint8* buffer, int bufferSize, const hknpShapeKeyMask* mask ) const
{
	HK_ERROR( 0xaf6a7712, "createShapeKeyIterator() for this shape type not available on SPU." );
	return HK_NULL;
}

#endif

#if !defined(HK_PLATFORM_SPU)
void hknpShape::getAllShapeKeys( const hknpShapeKeyPath& shapeKeyPath, const hknpShapeKeyMask* mask, hkFixedCapacityArray<hknpShapeKeyPath>* keyPathsOut ) const
#else
void hknpShape::getAllShapeKeys( const hknpShapeKeyPath& shapeKeyPath, const hknpShapeKeyMask* mask, hkUint8* shapeBuffer, int shapeBufferSize, hkFixedCapacityArray<hknpShapeKeyPath>* keyPathsOut ) const
#endif
{
	HKNP_SHAPE_NOT_IMPLEMENTED(0x47b26ded);
}

void hknpShape::getLeafShape( hknpShapeKey key, hknpShapeCollector* collector ) const
{
	HKNP_SHAPE_NOT_IMPLEMENTED(0x1d729a38);
}

void hknpShape::queryAabbImpl(
	hknpCollisionQueryContext* queryContext, const hknpAabbQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
	const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
	hkArray<hknpShapeKey>* hits, hknpQueryAabbNmp* nmpInOut ) const
{
	HKNP_SHAPE_NOT_IMPLEMENTED(0x698a584c);
}

void hknpShape::queryAabbImpl(
	hknpCollisionQueryContext* queryContext, const hknpAabbQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
	const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
	hknpCollisionQueryCollector* collector, hknpQueryAabbNmp* nmpInOut ) const
{
	HKNP_SHAPE_NOT_IMPLEMENTED(0x249de87e);
}

void hknpShape::castRayImpl(
	hknpCollisionQueryContext* queryContext, const hknpRayCastQuery& query,
	const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
	hknpCollisionQueryCollector* collector ) const
{
	HKNP_SHAPE_NOT_IMPLEMENTED(0x7ba2ada7);
}

void hknpShape::getSignedDistances( const hknpShape::SdfQuery& query, hknpShape::SdfContactPoint* contactsOut ) const
{
	HKNP_SHAPE_NOT_IMPLEMENTED(0x168b0b4f);
}

#if defined(HK_PLATFORM_SPU)
int hknpShape::getSignedDistanceContacts(
	const hknpSimulationThreadContext& tl, const hknpShape* queryShape, const hkTransform& sdfFromQueryTransform, hkReal maxDistance,
	int vertexIdOffset, SdfContactPointWriter& contactPointsOut ) const
{
	HKNP_SHAPE_NOT_IMPLEMENTED(0x3aafa437);
	return 0;
}
#endif

int hknpShape::getNumberOfSupportVertices() const
{
	HKNP_SHAPE_NOT_IMPLEMENTED(0x2c82390a);
	return 0;
}

const hkcdVertex* hknpShape::getSupportVertices( hkcdVertex* vertexBuffer, int bufferSize ) const
{
	HKNP_SHAPE_NOT_IMPLEMENTED(0x6c22654c);
	return HK_NULL;
}

void hknpShape::getSupportingVertex( hkVector4Parameter direction, hkcdVertex* vertexBufferOut ) const
{
	HKNP_SHAPE_NOT_IMPLEMENTED(0x28d78a7d);
}

void hknpShape::convertVertexIdsToVertices( const hkUint8* ids, int numIds, hkcdVertex* verticesOut ) const
{
	HKNP_SHAPE_NOT_IMPLEMENTED(0x43342a57);
}

int hknpShape::getNumberOfFaces() const
{
	HKNP_SHAPE_NOT_IMPLEMENTED(0x7205d788);
	return 0;
}

int hknpShape::getFaceVertices( const int faceIndex, hkVector4& planeOut, hkcdVertex* vertexBufferOut ) const
{
	HKNP_SHAPE_NOT_IMPLEMENTED(0x480c7f47);
	return 0;
}

void hknpShape::getFaceInfo( const int faceIndex, hkVector4& planeOut, int& minAngleOut ) const
{
	HKNP_SHAPE_NOT_IMPLEMENTED(0x7d77da16);
}

int hknpShape::getSupportingFace(
	hkVector4Parameter surfacePoint, const hkcdGsk::Cache* gskCache, bool useB,
	hkVector4& planeOut, int& minAngleOut, hkUint32 &prevFaceId ) const
{
	HKNP_SHAPE_NOT_IMPLEMENTED(0xa03840a);
	return 0;
}

hkReal hknpShape::calcMinAngleBetweenFaces() const
{
	HKNP_SHAPE_NOT_IMPLEMENTED(0x5d043200);
	return 0.0f;
}


#if !defined(HK_PLATFORM_SPU)

hknpShapeMassProperties::hknpShapeMassProperties( hkFinishLoadedObjectFlag flag )
	: hkReferencedObject( flag )
{
}

hknpShape::hknpShape( hkFinishLoadedObjectFlag flag )
	:	hkReferencedObject(flag)
{
}

hknpShape::~hknpShape()
{
	if ( m_properties && m_memSizeAndFlags )
	{
		delete m_properties;
	}
	m_properties = HK_NULL;
}

hknpShape::MutationSignals* hknpShape::getMutationSignals()
{
	return HK_NULL;
}

hknpShapeKeyMask* hknpShape::createShapeKeyMask() const
{
	HKNP_SHAPE_NOT_IMPLEMENTED(0xC26A7711);
	return HK_NULL;
}

#else

void hknpShape::patchShapeKeyMaskVTable( hknpShapeKeyMask* mask ) const
{
	HKNP_SHAPE_NOT_IMPLEMENTED(0xafe1417f);
}

#endif

HK_AUTO_INLINE int hknpShape::getSignedDistanceContactsImpl(
	const hknpSimulationThreadContext& tl, const hknpShape* queryShape, const hkTransform& sdfFromQueryTransform,
	hkReal maxDistance, int vertexIdOffset, SdfContactPointWriter& contactPointsOut ) const
{
	if ( queryShape->m_dispatchType == hknpCollisionDispatchType::CONVEX )
	{
		int numVerts = queryShape->getNumberOfSupportVertices();
		hkcdVertex* sCenterBuffer =	 hkAllocateStack<hkcdVertex>(numVerts, "sdfSourceVertexBuffer");
		hknpShape::SdfContactPoint* verts = hkAllocateStack<hknpShape::SdfContactPoint>(numVerts, "sdfResultBuffer");

		const hkcdVertex* centers = queryShape->getSupportVertices( sCenterBuffer, numVerts );
		hkVector4Util::transformPoints( sdfFromQueryTransform, centers, numVerts, sCenterBuffer );

		hknpShape::SdfQuery query;
		query.m_maxDistance = HK_REAL_MAX;
		query.m_spheresRadius = queryShape->m_convexRadius;
		query.m_numSpheres = numVerts;
		query.m_sphereCenters = (hkVector4*)sCenterBuffer;
		getSignedDistances( query, verts );

		hkSimdReal radius; radius.setFromFloat( queryShape->m_convexRadius );

		for (int i=0; i < numVerts; i++)
		{
			const hknpShape::SdfContactPoint& v = verts[i];
			if ( v.m_distance <= maxDistance )
			{
				hknpShape::SdfContactPoint* HK_RESTRICT cp = contactPointsOut.reserve( sizeof(hknpShape::SdfContactPoint) );
				hkString::memCpy16<sizeof(hknpShape::SdfContactPoint) >( cp, &v );

				hkSimdReal distance = hkSimdReal::fromFloat(v.m_distance);
				cp->m_vertexId = hknpVertexId(vertexIdOffset) + v.m_vertexId;
				distance.store<1>( &cp->m_distance );
				contactPointsOut.advance( sizeof(hknpShape::SdfContactPoint) );
			}
		}
		hkDeallocateStack(verts, numVerts);
		hkDeallocateStack(sCenterBuffer, numVerts);
		return numVerts;
	}
	else
	{
		HK_ASSERT2( 0xf034cdff, queryShape->m_dispatchType == hknpCollisionDispatchType::COMPOSITE,
			"Distance field shapes can only collide with convex or composite shapes" );

#if !defined(HK_PLATFORM_SPU)

		hknpShapeCollectorWithInplaceTriangle leafShapeCollector;
		int indexOffset = vertexIdOffset;
		for( hkRefPtr<hknpShapeKeyIterator> it = queryShape->createShapeKeyIterator(); it->isValid(); it->next() )	
		{
			leafShapeCollector.reset( sdfFromQueryTransform );
			queryShape->getLeafShape( it->getKey(), &leafShapeCollector );
			indexOffset += getSignedDistanceContacts(
				tl, leafShapeCollector.m_shapeOut, leafShapeCollector.m_transformOut, maxDistance, indexOffset, contactPointsOut );
		}

#else

		hknpShapeCollector leafShapeCollector( tl.m_triangleShapePrototypes[0] );

		int indexOffset = vertexIdOffset;
		{
			hknpShapeKeyPath root;
			const int shapeBufferSize = 2 * HKNP_SHAPE_BUFFER_SIZE;
			HK_ALIGN16( hkUint8 ) shapeBuffer[ shapeBufferSize ];
			hkFixedCapacityArray<hknpShapeKeyPath> shapeKeyPaths( 256, "shapeKeyPaths" ); 
			queryShape->getAllShapeKeys( root, HK_NULL, shapeBuffer, shapeBufferSize, &shapeKeyPaths );

			for ( int i = 0; i < shapeKeyPaths.getSize(); i++ )
			{
				leafShapeCollector.reset( sdfFromQueryTransform );
				queryShape->getLeafShape( shapeKeyPaths[i].getKey(), &leafShapeCollector );
				indexOffset += getSignedDistanceContacts(
					tl, leafShapeCollector.m_shapeOut, leafShapeCollector.m_transformOut, maxDistance, indexOffset, contactPointsOut );
			}

			shapeKeyPaths.clearAndDeallocate();
		}

#endif

		return indexOffset - vertexIdOffset;
	}
}

#if !defined(HK_PLATFORM_SPU)

int hknpShape::getSignedDistanceContacts(
	const hknpSimulationThreadContext& tl, const hknpShape* queryShape, const hkTransform& sdfFromQueryTransform, hkReal maxDistance,
	int vertexIdOffset, SdfContactPointWriter& contactPointsOut ) const
{
	return getSignedDistanceContactsImpl( tl, queryShape, sdfFromQueryTransform, maxDistance, vertexIdOffset, contactPointsOut );
}

void hknpShape::buildMassProperties( const MassConfig& massConfig, hkDiagonalizedMassProperties& massPropertiesOut ) const
{
	HK_WARN_ON_DEBUG_IF( massConfig.m_quality == MassConfig::QUALITY_HIGH, 0xC26A7712,
		"Requested high quality mass properties, but using AABB approximation." );

	hkAabb aabb;
	calcAabb( hkTransform::getIdentity(), aabb );

	hknpShapeUtil::buildAabbMassProperties( massConfig, aabb, massPropertiesOut );
}

hkResult hknpShape::buildSurfaceGeometry( const BuildSurfaceGeometryConfig& config, hkGeometry* geometryOut ) const
{
	return HK_FAILURE;
}

void hknpShape::checkConsistency() const
{

}

#endif // !defined(HK_PLATFORM_SPU)


#if defined( HK_PLATFORM_PPU )

//
//	Automatically set the SPU flags on this shape.

void hknpShape::computeSpuFlags()
{
	const int shapeSize = calcSize();
	if ( shapeSize > HKNP_MAX_SHAPE_SIZE_ON_SPU )
	{
		m_flags.orWith(SHAPE_NOT_SUPPORTED_ON_SPU);
	}
}

#endif

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
