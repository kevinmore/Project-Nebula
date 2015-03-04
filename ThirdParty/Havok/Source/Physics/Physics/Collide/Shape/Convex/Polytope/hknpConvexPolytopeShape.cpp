/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Collide/Shape/Convex/Polytope/hknpConvexPolytopeShape.h>

#include <Common/Internal/GeometryProcessing/CollisionGeometryOptimizer/hkgpCgo.h>

#include <Geometry/Internal/Algorithms/Distance/hkcdDistancePointConvex.h>

#if !defined(HK_PLATFORM_SPU)
#	include <Common/Internal/GeometryProcessing/ConvexHull/hkgpConvexHull.h>
#	include <Common/GeometryUtilities/Inertia/hkInertiaTensorComputer.h>
#	include <Physics/Physics/Dynamics/Motion/hknpMotionUtil.h>
#endif


#if !defined(HK_PLATFORM_SPU)

hknpConvexPolytopeShape::hknpConvexPolytopeShape( hkFinishLoadedObjectFlag flag )
	:	hknpConvexShape(flag)
{
	if( flag.m_finishing )
	{
		m_flags.orWith( IS_CONVEX_POLYTOPE_SHAPE );
	}
}

#endif // !defined(HK_PLATFORM_SPU)

int hknpConvexPolytopeShape::calcSize() const
{
	// Check alignment of each relarray's data
	HK_ON_DEBUG( const int mask = HK_REAL_ALIGNMENT - 1; )
	HK_ASSERT(0x2324fc46, ((hkUlong)(HK_OFFSET_OF(hknpConvexPolytopeShape, m_vertices) + m_vertices.getOffset()) & mask) == 0);
	HK_ASSERT(0x2324fc47, ((hkUlong)(HK_OFFSET_OF(hknpConvexPolytopeShape, m_planes) + m_planes.getOffset()) & mask) == 0);
	HK_ASSERT(0x2324fc48, ((hkUlong)(HK_OFFSET_OF(hknpConvexPolytopeShape, m_faces) + m_faces.getOffset()) & mask) == 0);
	HK_ASSERT(0x2324fc49, ((hkUlong)(HK_OFFSET_OF(hknpConvexPolytopeShape, m_indices) + m_indices.getOffset()) & mask) == 0);

	// Since the relarray datas may be in any order (due to serialization), we look for the end of the shape in memory
	// by finding the maximum offset from the start of the shape to the end of each relarray's data.
	const int shapeSize		= sizeof(hknpConvexPolytopeShape);
	const int verticesSize	= HK_OFFSET_OF(hknpConvexPolytopeShape, m_vertices) + m_vertices.getOffset() + m_vertices.getSize() * sizeof(hkVector4);
	const int planesSize	= HK_OFFSET_OF(hknpConvexPolytopeShape, m_planes) + m_planes.getOffset() + m_planes.getSize() * sizeof(hkVector4);
	const int facesSize		= HK_OFFSET_OF(hknpConvexPolytopeShape, m_faces) + m_faces.getOffset() + m_faces.getSize() * sizeof(Face);
	const int indicesSize	= HK_OFFSET_OF(hknpConvexPolytopeShape, m_indices) + m_indices.getOffset() + m_indices.getSize() * sizeof(VertexIndex);

	const int maxAB = hkMath::max2(verticesSize, planesSize);
	const int maxCD = hkMath::max2(facesSize, indicesSize);
	const int maxABCD = hkMath::max2(maxAB, maxCD);
	const int size = HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, hkMath::max2(shapeSize, maxABCD));

	HK_ASSERT2( 0x3be399a5, size == calcConvexPolytopeShapeSize(getNumberOfVertices(), getNumFaces(), getNumIndices()),
		"Calculated convex shape size is different to the expected size" );

	return size;
}

void hknpConvexPolytopeShape::getSignedDistances( const hknpShape::SdfQuery& query, SdfContactPoint* contactsOut ) const
{
	for (int i =0; i < query.m_numSpheres; i++)
	{
		hknpShape::SdfContactPoint* HK_RESTRICT cp = &contactsOut[i];
		hkSimdReal distance;
		hkVector4  pos;
		hkcdDistancePointConvex::hkcdPointConvexApprox( query.m_sphereCenters[i], getPlanes(), getNumFaces(), pos, cp->m_normal, distance );

		hkSimdReal sphereRadius; sphereRadius.setFromFloat(query.m_spheresRadius);
		distance = distance-(hkSimdReal::fromFloat(m_convexRadius)-sphereRadius);
		cp->m_vertexId  = hknpVertexId(i);
		cp->m_shapeKey  = HKNP_INVALID_SHAPE_KEY;
		cp->m_shapeTag  = HKNP_INVALID_SHAPE_TAG;
		cp->m_position.setAddMul( pos, cp->m_normal, sphereRadius );
		distance.store<1>( &cp->m_distance );
	}
}


#if !defined(HK_PLATFORM_SPU)

static int reduceEdgesInFace( int numVerticesPerFace, hkVector4* vPtr, hkArray<hknpConvexPolytopeShape::VertexIndex>& reducedIndices )
{
	// reduce our face
	while ( numVerticesPerFace > hknpConvexPolytopeShape::MAX_NUM_VERTICES_PER_FACE)
	{
		// search for the smallest triangle
		hkSimdReal m = hkSimdReal_Max;
		int index = 0;
		for (int vertexIndex=0; vertexIndex < numVerticesPerFace; vertexIndex++ )
		{
			int p = (vertexIndex==0)? (numVerticesPerFace-1) : vertexIndex-1;
			int n = (vertexIndex+1)%numVerticesPerFace;
			hkVector4 pp = vPtr[reducedIndices[p]];
			hkVector4 pi = vPtr[reducedIndices[vertexIndex]];
			hkVector4 pn = vPtr[reducedIndices[n]];
			pp.sub(pi);
			pn.sub(pi);
			hkVector4 normal; normal.setCross( pp, pn );
			hkSimdReal areaSqrd = normal.lengthSquared<3>();
			if ( areaSqrd < m )
			{
				m = areaSqrd;
				index = vertexIndex;
			}
		}
		// remove smallest point attached to smallest triangle
		numVerticesPerFace--;
		reducedIndices.removeAtAndCopy(index);
	}
	return numVerticesPerFace;
}

// Find a face from a given edge.
static int findFace(int a,int b, const hkArray<int>& vpf, const int* indices)
{
	for(int i=0;i<vpf.getSize();++i)
	{
		const int numIndices = vpf[i];
		for(int j=numIndices-1,k=0;k<numIndices;j=k++)
		{
			if(indices[j]==a && indices[k]==b) return i;
		}
		indices	+= numIndices;
	}
	return -1;
}

hknpConvexShape* hknpConvexPolytopeShape::createFromVerticesInternal(
	const hkStridedVertices& verticesIn, hkReal radius, const BuildConfig& config )
{
	if ( (verticesIn.getSize() <= 0) || (radius < 0.0f) )
	{
		HK_WARN( 0x5B234D88, "No vertices / negative convex radius!" );
		return HK_NULL;
	}

	// Create a convex hull from the input data
	hkgpConvexHull hull;
	hkgpConvexHull::BuildConfig hullBuildConfig;
	{
		hullBuildConfig.m_buildMassProperties = config.m_buildMassProperties;
		HK_ON_DEBUG( hullBuildConfig.m_buildMassProperties = true );	// this is necessary to allow for extra checks

		const int dim = hull.build( verticesIn, hullBuildConfig );
		if( dim < 2 )
		{
			HK_WARN(0x5B234D88, "Face information cannot be built for degenerated convex hulls.");
			return HK_NULL;
		}

		HK_ASSERT2( 0x8a4d7c5, hull.getNumVertices() <= verticesIn.getSize(),
			"Generated hull has more vertices than the input" );
	}

	// Build mass properties (optional)
	hkRefPtr<hknpShapeMassProperties> shapeMassProperties = HK_NULL;
	if( config.m_buildMassProperties && hull.hasValidMassProperties() )
	{
		hkMassProperties massProperties;
		massProperties.m_mass = 1.0f;

		if( config.m_shrinkByRadius || (radius == 0.0f) )
		{
			// Take properties straight from the hull
			massProperties.m_volume = hull.getVolume().getReal();
			massProperties.m_centerOfMass = hull.getCenterOfMass();
			massProperties.m_inertiaTensor = hull.getWorldInertia();
		}
		else
		{
			// Need to expand the hull first
			hkgpConvexHull* HK_RESTRICT expandedHull = hull.clone();
			expandedHull->absoluteScale( radius );
			massProperties.m_volume = expandedHull->getVolume().getReal();
			massProperties.m_centerOfMass = expandedHull->getCenterOfMass();
			massProperties.m_inertiaTensor = expandedHull->getWorldInertia();
			expandedHull->removeReference();
		}

		massProperties.m_mass = config.m_massConfig.calcMassFromVolume( massProperties.m_volume );

		hkSimdReal inertiaFactor;
		inertiaFactor.setFromFloat( massProperties.m_mass * config.m_massConfig.m_inertiaFactor );
		massProperties.m_inertiaTensor.mul( inertiaFactor );

		shapeMassProperties.setAndDontIncrementRefCount(new hknpShapeMassProperties);
		shapeMassProperties->m_compressedMassProperties.pack( massProperties );
	}

	// Shrink and simplify the hull (optional)
	{
		// Shrink the hull by the requested radius.
		// NOTE: This can change the value of the requested radius.
		if( config.m_shrinkByRadius && (radius > 0.0f) )
		{
			hkgpConvexHull::AbsoluteScaleConfig	scaleConfig;
			scaleConfig.m_method = hkgpConvexHull::AbsoluteScaleConfig::SKM_CLAMPED_PLANES;
			scaleConfig.m_featurePreservationFactor = config.m_featurePreservationFactor;
			radius = -hull.absoluteScale( -radius, scaleConfig );

			if ( radius < 0.0f )
			{
				HK_WARN( 0x2693cee5, "Negative convex radius after hull shrinking" );
				return HK_NULL;
			}
			if( hull.getNumVertices() > verticesIn.getSize() )
			{
				HK_WARN( 0x5B234D88, "Shrunken hull has an increased number of vertices!" );
				return HK_NULL;	
			}
		}

		// Optionally simplify again using geometry simplifier
		
		if( config.m_simplifyExpansionDistance > 0.0f )
		{
			hkGeometry geometry;
			hull.generateGeometry( hkgpConvexHull::INTERNAL_VERTICES, geometry, 0 );

			hkgpCgo::Config cgoConfig;
			cgoConfig.m_maxDistance = config.m_simplifyExpansionDistance;
			//config.m_maxShrink = 0.2f;
			hkgpCgo::optimize( cgoConfig, geometry );

			hull.build( geometry.m_vertices, hullBuildConfig );
		}

		// Simplify to required max number of vertices
		{
			HK_ON_DEBUG( const hkReal baseSurfaceArea = hull.getSurfaceArea().getReal() );
			hkgpConvexHull::SimplifyConfig simplifyConfig;
			simplifyConfig.m_maxVertices = hkMath::min2((hkUint8)MAX_NUM_VERTICES, config.m_maxNumVertices);
			simplifyConfig.m_removeUnreferencedVertices = true;

			#if 1 
			hull.decimateVertices( simplifyConfig.m_maxVertices, simplifyConfig.m_ensureContainment, true );
			#else
			if ( hull.simplify( simplifyConfig ) != HK_SUCCESS )
			{
				return HK_NULL;
			}
			#endif

			HK_ASSERT2( 0x127ac5c, hull.getNumVertices() <= simplifyConfig.m_maxVertices,
				"Simplified hull has more vertices than were requested" );

			HK_ON_DEBUG( const hkReal finalSurfaceArea = hull.getSurfaceArea().getReal() );
			HK_ON_DEBUG( const hkReal saRatio = hkMath::abs( (finalSurfaceArea-baseSurfaceArea) / baseSurfaceArea ) );
			HK_WARN_ON_DEBUG_IF( saRatio >= 0.1f, 0x0F6E081BF, "Simplification has introduced an surface area error of "
				<< (int)(saRatio * hkReal(100.0f) + hkReal(0.5f)) << "%" );
		}
	}

	// Construct a shape from the hull
	hknpConvexPolytopeShape* shape = HK_NULL;
	{
		// Extract hull data
		hkArray<hkVector4> vertices;
		hkArray<int> verticesPerFace, indices;
		hkArray<hkVector4> planes;
		hull.fetchPositions( hkgpConvexHull::INTERNAL_VERTICES, vertices );
		hull.generateIndexedFaces( hkgpConvexHull::INTERNAL_VERTICES, verticesPerFace, indices, true );
		hull.fetchPlanes( planes );

		if ( planes.getSize() != verticesPerFace.getSize() )
		{
			HK_WARN( 0x5B234D88, "Inconsistent planes!" );
			return HK_NULL;
		}

		// Index the vertices
		for( int i=0; i<vertices.getSize(); ++i )
		{
			vertices[i].setInt24W(i);
		}

		// Pad the vertices
		while( vertices.getSize() & 3 )
		{
			hkVector4 v = vertices.back();
			vertices.pushBack(v);
		}

		// Pad the planes
		const int numFaces = verticesPerFace.getSize();
		const int numFacesMin4 = hkMath::max2( numFaces, 4 );
		while( verticesPerFace.getSize() & 3 )
		{
			planes.expandOne().set( 0, 0, 0, -HK_REAL_MAX );
			verticesPerFace.pushBack(0);
		}

		const int numIndices	= indices.getSize();
		const int numVertices	= vertices.getSize();
		const int numPlanes		= planes.getSize();

		// Construct shape
		{
			int shapeSize;
			void* buffer = allocateConvexPolytopeShape(
				numVertices, numFacesMin4, numIndices, config.m_sizeOfBaseClass, shapeSize );
		#if defined(HK_PLATFORM_HAS_SPU)
			if ( shapeSize > HKNP_MAX_SHAPE_SIZE_ON_SPU )
			{
				HK_WARN(0x5849d4cd, "Shape exceeds maximum allowed size on SPU");
			}
		#endif
			shape = new (buffer) hknpConvexPolytopeShape(
				numVertices, numFacesMin4, numIndices, radius, config.m_sizeOfBaseClass );
			shape->m_memSizeAndFlags = (hkUint16) shapeSize;
		}

		hkVector4*		vtxPtrOut	= shape->getVertices();
		hkVector4*		planePtrOut	= shape->getPlanes();
		Face*			facesPtrOut	= shape->getFaces();
		VertexIndex*	fiPtrOut	= shape->getIndices();

		// Build vertices
		for( int i=0; i<numVertices; i++ )
		{
			vtxPtrOut[i] = vertices[i];
		}

		// Build planes
		for( int i=0; i<numPlanes; i++ )
		{
			planePtrOut[i] = planes[i];
		}

		// Build faces and indices
		{
			const int* indicesIn = indices.begin();
			for( int faceIndex = 0; faceIndex<numFaces; ++faceIndex )
			{
				int numVerticesPerFace = verticesPerFace[faceIndex];
				facesPtrOut[faceIndex].m_firstIndex	= (hkUint16)( fiPtrOut - shape->getIndices() );

				if( numVerticesPerFace > MAX_NUM_VERTICES_PER_FACE )
				{
					// reduce the face edges
					hkInplaceArray<VertexIndex, MAX_NUM_VERTICES_PER_FACE * 4> reducedIndices;
					reducedIndices.setSize( numVerticesPerFace );
					{
						for( int j=0; j<numVerticesPerFace; ++j )
						{
							reducedIndices[j] = (VertexIndex) indicesIn[j];
						}
						indicesIn += numVerticesPerFace;
					}
					numVerticesPerFace = reduceEdgesInFace( numVerticesPerFace, vtxPtrOut, reducedIndices );

					for( int j=0; j<numVerticesPerFace; ++j )
					{
						*fiPtrOut++ = reducedIndices[j];
					}
				}
				else
				{
					for( int j=0; j<numVerticesPerFace; ++j )
					{
						*fiPtrOut++ = (VertexIndex) indicesIn[j];
					}
					indicesIn += numVerticesPerFace;
				}

				facesPtrOut[faceIndex].m_numIndices	= (hkUint8)numVerticesPerFace;
			}
			{
				// make sure we have at least 4 faces
				for (int faceIndex = numFaces; faceIndex < numFacesMin4; faceIndex++)
				{
					facesPtrOut[faceIndex] = facesPtrOut[0];
				}
			}
		}

		// Number of indices may have been reduced
		shape->m_indices._setSize( hkUint16(fiPtrOut - shape->getIndices() ));

		// Compute minimum edge angle per face
		const int* baseIndices( indices.begin() );
		int maxNumEdgesPerFace = 0;
		for( int i=0; i<numFaces; ++i )
		{
			const int numInd = verticesPerFace[i];
			maxNumEdgesPerFace = hkMath::max2( maxNumEdgesPerFace, numInd );
			hkReal maxCosAngle = 0;
			for( int j=numInd-1, k=0; k<numInd; j=k++ )
			{
				const int f = findFace( baseIndices[k], baseIndices[j], verticesPerFace, indices.begin() );
				maxCosAngle = hkMath::max2( maxCosAngle, planes[i].dot<3>( planes[f] ).getReal() );
			}
			facesPtrOut[i].m_minHalfAngle = (hkUint8)hknpMotionUtil::convertAngleToAngularTIM( hkMath::acos(maxCosAngle)*0.5f );
			baseIndices += numInd;
		}

		// Enable bplane collisions?
		// The current bplane-algorithm can handle only faces with 4 edges
		if( maxNumEdgesPerFace <= 4)
		{
			shape->setFlags( shape->getFlags().get() | hknpShape::SUPPORTS_BPLANE_COLLISIONS );
		}
	}

	// Attach mass properties
	if( config.m_buildMassProperties )
	{
		if( !shapeMassProperties )
		{
			// Building failed, so fall back to base AABB method
			hkDiagonalizedMassProperties massProperties;
			shape->hknpShape::buildMassProperties( config.m_massConfig, massProperties );

			shapeMassProperties.setAndDontIncrementRefCount(new hknpShapeMassProperties);
			shapeMassProperties->m_compressedMassProperties.pack( massProperties );
		}

		shape->setProperty( hknpShapePropertyKeys::MASS_PROPERTIES, shapeMassProperties );
	}

	return shape;
}

#endif // !defined(HK_PLATFORM_SPU)


hkReal hknpConvexPolytopeShape::calcMinAngleBetweenFaces() const
{
	int minHalfAngle = 255; // 255 = 90deg
	for( int i = 0; i < getNumFaces(); i++ )
	{
		hkVector4 plane;
		int angle;
		getFaceInfo(i, plane, angle);
		minHalfAngle = hkMath::min2(minHalfAngle, angle);
	}

	return HK_REAL_PI / 510.0f * 2.0f * minHalfAngle;
}

int hknpConvexPolytopeShape::getFaceVertices(
	const int faceIndex, hkVector4& planeOut, hkcdVertex* const HK_RESTRICT vertexBuffer ) const
{
	static const hknpConvexPolytopeShape::VertexIndex mo = hknpConvexPolytopeShape::VertexIndex(~0);
	static const hknpConvexPolytopeShape::VertexIndex indexMask[] = {
		mo, mo, mo, mo, mo, mo, mo, mo, mo, mo, mo, mo, mo, mo, mo, mo,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	HK_COMPILE_TIME_ASSERT( sizeof(indexMask) >= hknpConvexPolytopeShape::MAX_NUM_VERTICES_PER_FACE * sizeof(hknpConvexPolytopeShape::VertexIndex)*2 );

	// Get vertices from shape
	const hkcdVertex* vertices = getVertices();
	const hknpConvexPolytopeShape::Face &face = getFace( faceIndex );
	const int numVertices = face.m_numIndices;
	const hknpConvexPolytopeShape::VertexIndex* indexMasks = &indexMask[ 16 - numVertices ];
	const hknpConvexPolytopeShape::VertexIndex* indices	= &getIndices()[face.m_firstIndex];

	// Write the plane
	planeOut = m_planes[faceIndex];

	hkcdVertex v0; v0.load<4>( &vertices[indices[/*indexMasks[0]&*/(0)]](0) ); // a face has at least 2 edges, no need to check them
	hkcdVertex v1; v1.load<4>( &vertices[indices[/*indexMasks[1]&*/(1)]](0) );
	hkcdVertex v2; v2.load<4>( &vertices[indices[indexMasks[2] & (2)]](0) );
	hkcdVertex v3; v3.load<4>( &vertices[indices[indexMasks[3] & (3)]](0) );
	vertexBuffer[0] = v0;
	vertexBuffer[1] = v1;
	vertexBuffer[2] = v2;
	vertexBuffer[3] = v3;

	if ( hknpConvexPolytopeShape::MAX_NUM_VERTICES_PER_FACE > 4 )
	{
		vertexBuffer[4].load<4>( &vertices[indices[indexMasks[4] & (4)]](0) );
		vertexBuffer[5].load<4>( &vertices[indices[indexMasks[5] & (5)]](0) );
		vertexBuffer[6].load<4>( &vertices[indices[indexMasks[6] & (6)]](0) );
		vertexBuffer[7].load<4>( &vertices[indices[indexMasks[7] & (7)]](0) );
	}

	if ( hknpConvexPolytopeShape::MAX_NUM_VERTICES_PER_FACE > 8 )
	{
		vertexBuffer[8].load<4>( &vertices[indices[indexMasks[8] & (8)]](0) );
		vertexBuffer[9].load<4>( &vertices[indices[indexMasks[9] & (9)]](0) );
		vertexBuffer[10].load<4>( &vertices[indices[indexMasks[10] & (10)]](0) );
		vertexBuffer[11].load<4>( &vertices[indices[indexMasks[11] & (11)]](0) );
	}

	if ( hknpConvexPolytopeShape::MAX_NUM_VERTICES_PER_FACE > 12 )
	{
		vertexBuffer[12].load<4>( &vertices[indices[indexMasks[12] & (12)]](0) );
		vertexBuffer[13].load<4>( &vertices[indices[indexMasks[13] & (13)]](0) );
		vertexBuffer[14].load<4>( &vertices[indices[indexMasks[14] & (14)]](0) );
		vertexBuffer[15].load<4>( &vertices[indices[indexMasks[15] & (15)]](0) );
	}

	return numVertices;
}

#if !defined(HK_PLATFORM_SPU)

void hknpConvexPolytopeShape::checkConsistency() const
{
	HK_ASSERT( 0xf0dc5c00,    (void*)m_vertices.begin()>= (void*)(this+1) );
	HK_ASSERT( 0xf0dc5c01,    (void*)m_planes.begin()  >= (void*)&m_vertices[ m_vertices.getSize()]  );
	HK_ASSERT( 0xf0dc5c02,    (void*)m_faces.begin()   >= (void*)&m_planes[ m_planes.getSize()]  );
	HK_ASSERT( 0xf0dc5c03,    (void*)m_indices.begin() >= (void*)&m_faces[ m_faces.getSize()]  );
	HK_ASSERT( 0xf0dc5c04,    (void*)&m_faces[ m_faces.getSize()] >= hkAddByteOffsetConst(this, this->getAllocatedSize() ) || 0==this->getAllocatedSize());

	// check vertices
	for (int i =0; i < getNumberOfVertices(); i++)
	{
		HK_ASSERT( 0xf0dc5c10, getVertex(i).isOk<4>() );
		HK_ASSERT( 0xf0dc5c11, getVertex(i).getInt16W() <=i  );
	}

	// check planes
	HK_ASSERT( 0xf0dc5c20, m_planes.getSize() == m_faces.getSize() );
	for (int i =0; i < getNumFaces(); i++)
	{
		HK_ASSERT( 0xf0dc5c21, m_planes[i].isOk<4>() );
	}
}

#endif

/*
 * Havok SDK - Product file, BUILD(#20130912)
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
