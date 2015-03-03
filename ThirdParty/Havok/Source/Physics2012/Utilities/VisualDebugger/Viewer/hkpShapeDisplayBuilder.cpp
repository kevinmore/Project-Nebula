/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Common/Base/Types/Geometry/hkGeometry.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/hkpShapeDisplayBuilder.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>

#include <Physics2012/Collide/Shape/Convex/Sphere/hkpSphereShape.h>
#include <Physics2012/Collide/Shape/Deprecated/MultiSphere/hkpMultiSphereShape.h>
#include <Physics2012/Collide/Shape/Convex/Box/hkpBoxShape.h>
#include <Physics2012/Collide/Shape/Convex/Cylinder/hkpCylinderShape.h>
#include <Physics2012/Collide/Shape/Misc/Transform/hkpTransformShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexTranslate/hkpConvexTranslateShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexTransform/hkpConvexTransformShape.h>
#include <Physics2012/Collide/Shape/Misc/Bv/hkpBvShape.h>
#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppBvTreeShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/hkpShapeCollection.h>
#include <Physics2012/Collide/Shape/Compound/Collection/List/hkpListShape.h>
#include <Physics2012/Collide/Shape/Convex/Triangle/hkpTriangleShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesShape.h>
#include <Physics2012/Collide/Shape/HeightField/SampledHeightField/hkpSampledHeightFieldShape.h>
#include <Physics2012/Collide/Shape/HeightField/Plane/hkpPlaneShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/ExtendedMeshShape/hkpExtendedMeshShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/StorageExtendedMesh/hkpStorageExtendedMeshShape.h>
#include <Physics2012/Internal/Collide/StaticCompound/hkpStaticCompoundShape.h>
#include <Physics2012/Internal/Collide/BvCompressedMesh/hkpBvCompressedMeshShape.h>

#include <Physics2012/Collide/Shape/Convex/Capsule/hkpCapsuleShape.h>
#include <Physics2012/Collide/Shape/Misc/MultiRay/hkpMultiRayShape.h>

#include <Physics2012/Collide/Shape/Deprecated/ConvexPieceMesh/hkpConvexPieceShape.h>

#include <Physics2012/Collide/Shape/Deprecated/CompressedMesh/hkpCompressedMeshShapeBuilder.h>

#include <Common/Internal/ConvexHull/hkGeometryUtility.h>


#include <Common/Visualize/Shape/hkDisplayPlane.h>
#include <Common/Visualize/Shape/hkDisplaySphere.h>
#include <Common/Visualize/Shape/hkDisplayCylinder.h>
#include <Common/Visualize/Shape/hkDisplayCapsule.h>
#include <Common/Visualize/Shape/hkDisplayBox.h>
#include <Common/Visualize/Shape/hkDisplayConvex.h>
#include <Common/Visualize/hkDebugDisplay.h>

#include <Common/GeometryUtilities/Misc/hkGeometryUtils.h>

#include <Common/Serialize/Util/hkBuiltinTypeRegistry.h>
#include <Common/Base/Reflection/Registry/hkVtableClassRegistry.h>

#include <Physics2012/Utilities/VisualDebugger/Viewer/hkpShapeContinueData.h>

hkpShapeDisplayBuilder::hkpShapeDisplayBuilderEnvironment::hkpShapeDisplayBuilderEnvironment()
:	m_spherePhiRes(8),
	m_sphereThetaRes(8)
{
}



hkpShapeDisplayBuilder::hkpShapeDisplayBuilder(const hkpShapeDisplayBuilderEnvironment& env)
:	m_environment(env),
	m_currentGeometry(HK_NULL)
{
}


void hkpShapeDisplayBuilder::buildDisplayGeometries(		const hkpShape* shape,
														hkArray<hkDisplayGeometry*>& displayGeometries)
{
	hkTransform transform;
	transform.setIdentity();

	resetCurrentRawGeometry();
	displayGeometries.clear();

	buildShapeDisplay(shape, transform, displayGeometries);
}

void hkpShapeDisplayBuilder::buildShapeDisplayTriangleSubPartsCompress(const hkpExtendedMeshShape* extendedMeshShape,
																		const hkTransform& transform,
																		hkArray<hkDisplayGeometry*>& displayGeometries)
{
	// Continue adding to the current geometry, if there is one.
	hkDisplayGeometry *const displayGeom = getCurrentRawGeometry(displayGeometries);
	hkGeometry *const geom = displayGeom->getGeometry();
	HK_ASSERT(0x4b5bb14e, geom != HK_NULL);
	// We need to remap pointers to floats (vertex as stored in the mesh) to indices in the display geometry.
	// This is needed to reindex.
	typedef hkPointerMap<const hkReal*, int> IndexMapType;
	IndexMapType indexMap;

	const int numTriangleSubparts = extendedMeshShape->getNumTrianglesSubparts();
	for ( int i = 0; i < numTriangleSubparts; ++i )
	{
		const hkpExtendedMeshShape::TrianglesSubpart& triangleSubpart = extendedMeshShape->getTrianglesSubpartAt( i );

		const hkVector4& extrusion = triangleSubpart.m_extrusion;
		
		// We share extruded vertices within triangle sub parts.
		IndexMapType extrusionMap;

		const int numTriangles = triangleSubpart.m_numTriangleShapes;
		for ( int j = 0; j < numTriangles; ++j )
		{
			const int triangleWindingFlip = j & triangleSubpart.m_flipAlternateTriangles;

			// The indices into the vertex array.
			int index[3];

			const void* data = (const void*) hkAddByteOffsetConst( triangleSubpart.m_indexBase, j * triangleSubpart.m_indexStriding );

			switch( triangleSubpart.m_stridingType )
			{
				case hkpExtendedMeshShape::INDICES_INT8:
				{
					const hkUint8* triangle = (const hkUint8*)data;

					index[0] = triangle[ 0 ];
					index[1] = triangle[ 1 + triangleWindingFlip ];
					index[2] = triangle[ 1 + (1 ^ triangleWindingFlip) ];
					break;
				}
				case hkpExtendedMeshShape::INDICES_INT16:
				{
					const hkUint16* triangle = (const hkUint16*)data;

					index[0] = triangle[ 0 ];
					index[1] = triangle[ 1 + triangleWindingFlip ];
					index[2] = triangle[ 1 + (1 ^ triangleWindingFlip) ];
					break;
				}
				case hkpExtendedMeshShape::INDICES_INT32:
				{
					const hkUint32* triangle = (const hkUint32*)data;

					index[0] = triangle[ 0 ];
					index[1] = triangle[ 1 + triangleWindingFlip ];
					index[2] = triangle[ 1 + (1 ^ triangleWindingFlip) ];
					break;
				}
				default:
				{
					// Initialize index vals to prevent 'uninitialized data' compiler warning.
					index[0] = 0;
					index[1] = 0;
					index[2] = 0;
					HK_ASSERT2( 0x12131a31,  triangleSubpart.m_stridingType == hkpExtendedMeshShape::INDICES_INT32, "Subpart index type is not set or out of range." );
				}
			}

			int newIndex[6];

			// Handle the original triangle
			for ( int vertNumber = 0; vertNumber < 3; ++vertNumber )
			{
				const hkReal *const oldVertex = hkAddByteOffsetConst( triangleSubpart.m_vertexBase, index[vertNumber] * triangleSubpart.m_vertexStriding );
				int potentialNewIndex = geom->m_vertices.getSize();
				IndexMapType::Iterator iterator = indexMap.findOrInsertKey( oldVertex, potentialNewIndex );
				newIndex[vertNumber] = indexMap.getValue( iterator );
				// The vertex wasn't found, so add it.
				if ( newIndex[vertNumber] == potentialNewIndex )
				{
					hkVector4& newVertex = geom->m_vertices.expandOne();
					newVertex.load<3,HK_IO_NATIVE_ALIGNED>( oldVertex );
					newVertex.setTransformedPos( triangleSubpart.m_transform, newVertex );
					newVertex.setTransformedPos( transform, newVertex );
				}
			}

			hkGeometry::Triangle& tri = *geom->m_triangles.expandBy(1);
			tri.set( newIndex[0], newIndex[1], newIndex[2] );

			// Handle extrusion
			if ( extrusion.lengthSquared<3>().isGreaterZero() )
			{
				// Write the vertices into the array, if necessary
				for ( int vertNumber = 3; vertNumber < 6; ++vertNumber )
				{
					const hkReal *const oldVertex = hkAddByteOffsetConst( triangleSubpart.m_vertexBase, index[vertNumber - 3] * triangleSubpart.m_vertexStriding );
					int potentialNewIndex = geom->m_vertices.getSize();
					IndexMapType::Iterator iterator = extrusionMap.findOrInsertKey( oldVertex, potentialNewIndex );
					newIndex[vertNumber] = extrusionMap.getValue( iterator );
					// The vertex wasn't found, so add it.
					if ( newIndex[vertNumber] == potentialNewIndex )
					{
						hkVector4& newVertex = geom->m_vertices.expandOne();
						newVertex.load<3,HK_IO_NATIVE_ALIGNED>( oldVertex );
						newVertex.setTransformedPos( triangleSubpart.m_transform, newVertex );
						newVertex.add( extrusion );
						newVertex.setTransformedPos( transform, newVertex );
					}
				}

				hkGeometry::Triangle* extrudedTris = geom->m_triangles.expandBy(7);
				extrudedTris[0].set( newIndex[3], newIndex[1], newIndex[0]);
				extrudedTris[1].set( newIndex[3], newIndex[4], newIndex[1]);
				extrudedTris[2].set( newIndex[4], newIndex[5], newIndex[1]);
				extrudedTris[3].set( newIndex[5], newIndex[2], newIndex[1]);
				extrudedTris[4].set( newIndex[5], newIndex[0], newIndex[2]);
				extrudedTris[5].set( newIndex[5], newIndex[3], newIndex[0]);
				extrudedTris[6].set( newIndex[5], newIndex[4], newIndex[3]);
			}
		}
	}
}


void hkpShapeDisplayBuilder::buildShapeDisplayTriangleSubPartsStorage(const hkpStorageExtendedMeshShape* extendedMeshShape,
																	   const hkTransform& transform,
																	   hkArray<hkDisplayGeometry*>& displayGeometries)
{
	// Continue adding to the current geometry, if there is one.
	hkDisplayGeometry *const displayGeom = getCurrentRawGeometry(displayGeometries);
	hkGeometry *const geom = displayGeom->getGeometry();
	HK_ASSERT(0x4b5bb14e, geom != HK_NULL);

	const int numTriangleSubparts = extendedMeshShape->getNumTrianglesSubparts();
	for ( int i = 0; i < numTriangleSubparts; ++i )
	{
		const hkpExtendedMeshShape::TrianglesSubpart& triangleSubpart = extendedMeshShape->getTrianglesSubpartAt( i );
		const hkpStorageExtendedMeshShape::MeshSubpartStorage& subpartStorage = *extendedMeshShape->m_meshstorage[i];

		const hkVector4& extrusion = triangleSubpart.m_extrusion;
		hkBool32 isExtruded = extrusion.lengthSquared<3>().isGreaterZero();

		HK_ASSERT2( 0x3d37a812, triangleSubpart.m_vertexBase == &subpartStorage.m_vertices.begin()[0](0), "Parts in storage extended mesh shape not compatible." );
		HK_ASSERT2( 0x3d37a813, triangleSubpart.m_vertexStriding == sizeof( hkVector4 ), "storage extended mesh shape with unexpected vertex striding" );

		const int numNewVertices = subpartStorage.m_vertices.getSize();
		
		// Copy the vertices into the geometry: A block of the original vertices, and then a block of their
		// extruded counterparts (if they are necessary).
		
		const int startOfNewIndices = geom->m_vertices.getSize();
		hkVector4* newVertex = geom->m_vertices.expandBy( numNewVertices * ( isExtruded ? 2 : 1 ) );
		for ( int j = 0; j < numNewVertices; ++j )
		{
			newVertex[j] = subpartStorage.m_vertices[j];
			newVertex[j].setTransformedPos( triangleSubpart.m_transform, newVertex[j] );
			newVertex[j].setTransformedPos( transform, newVertex[j] );
			if ( isExtruded )
			{
				hkVector4& newExtrudedVertex = newVertex[numNewVertices + j];
				newExtrudedVertex = subpartStorage.m_vertices[j];				
				newExtrudedVertex.setTransformedPos( triangleSubpart.m_transform, newExtrudedVertex );
				newExtrudedVertex.add( extrusion );
				newExtrudedVertex.setTransformedPos( transform, newExtrudedVertex );
			}
		}
		
		hkGeometry::Triangle* newTriangle = geom->m_triangles.expandBy( triangleSubpart.m_numTriangleShapes * ( isExtruded ? 8 : 1 ) );

		for ( int j = 0; j < triangleSubpart.m_numTriangleShapes; ++j )
		{
			int newIndex[3];

			const int triangleWindingFlip = j & triangleSubpart.m_flipAlternateTriangles;

			const void* data = (const void*) hkAddByteOffsetConst( triangleSubpart.m_indexBase, j * triangleSubpart.m_indexStriding );

			switch ( triangleSubpart.m_stridingType )
			{
				case hkpExtendedMeshShape::INDICES_INT8:
				{
					const hkUint8* triangle = static_cast<const hkUint8*>( data );

					newIndex[0] = startOfNewIndices + triangle[ 0 ];
					newIndex[1] = startOfNewIndices + triangle[ 1 + triangleWindingFlip ];
					newIndex[2] = startOfNewIndices + triangle[ 1 + (1 ^ triangleWindingFlip) ];
					break;
				}
				case hkpExtendedMeshShape::INDICES_INT16:
				{
					const hkUint16* triangle = static_cast<const hkUint16*>( data );

					newIndex[0] = startOfNewIndices + triangle[ 0 ];
					newIndex[1] = startOfNewIndices + triangle[ 1 + triangleWindingFlip ];
					newIndex[2] = startOfNewIndices + triangle[ 1 + (1 ^ triangleWindingFlip) ];
					break;
				}
				case hkpExtendedMeshShape::INDICES_INT32:
				{
					const hkUint32* triangle = static_cast<const hkUint32*>( data );

					newIndex[0] = startOfNewIndices + triangle[ 0 ];
					newIndex[1] = startOfNewIndices + triangle[ 1 + triangleWindingFlip ];
					newIndex[2] = startOfNewIndices + triangle[ 1 + (1 ^ triangleWindingFlip) ];
					break;
				}
				default:
				{
					// Suppress warning.
					newIndex[0] = 0;
					newIndex[1] = 0;
					newIndex[2] = 0;
					HK_ASSERT2( 0x12131a31,  triangleSubpart.m_stridingType == hkpExtendedMeshShape::INDICES_INT32, "Subpart index type is not set or out of range." );
				}
			}
			newTriangle[0].set( newIndex[0], newIndex[1], newIndex[2] );
			
			if ( isExtruded )
			{
				newTriangle[1].set( newIndex[0] + numNewVertices, newIndex[1]                 , newIndex[0] );
				newTriangle[2].set( newIndex[0] + numNewVertices, newIndex[1] + numNewVertices, newIndex[1] );
				newTriangle[3].set( newIndex[1] + numNewVertices, newIndex[2] + numNewVertices, newIndex[1] );
				newTriangle[4].set( newIndex[2] + numNewVertices, newIndex[2]                 , newIndex[1] );
				newTriangle[5].set( newIndex[2] + numNewVertices, newIndex[0]                 , newIndex[2] );
				newTriangle[6].set( newIndex[2] + numNewVertices, newIndex[0] + numNewVertices, newIndex[0] );
				newTriangle[7].set( newIndex[2] + numNewVertices, newIndex[1] + numNewVertices, newIndex[0] + numNewVertices);
				newTriangle += 7;
			}
		
			++newTriangle;
		}
	}
}

void hkpShapeDisplayBuilder::buildShapeDisplayShapeSubParts(const hkpExtendedMeshShape* extendedMeshShape,
															const hkTransform& transform,
															hkArray<hkDisplayGeometry*>& displayGeometries)
{
	const int numShapeSubparts = extendedMeshShape->getNumShapesSubparts();

	for ( int i = 0; i < numShapeSubparts; ++i )
	{
		const hkpExtendedMeshShape::ShapesSubpart& shapeSubpart = extendedMeshShape->getShapesSubpartAt( i );

		hkTransform subpartTransform;
		{
			hkTransform t; t.set( shapeSubpart.getRotation(), shapeSubpart.getTranslation() );
			subpartTransform.setMul( transform, t );
		}

		const int numChildShapes = shapeSubpart.m_childShapes.getSize();
		for ( int j = 0; j < numChildShapes; ++j )
		{
			buildShapeDisplay( shapeSubpart.m_childShapes[j], subpartTransform, displayGeometries );
		}
	}
}

void hkpShapeDisplayBuilder::buildShapeDisplay_ExtendedMeshShape( const hkpExtendedMeshShape* extendedMeshShape,
																   const hkTransform& transform,
																   hkArray<hkDisplayGeometry*>& displayGeometries )
{
	// Try to exploit the way the shape's data is stored in memory.
	if ( hkBuiltinTypeRegistry::getInstance().getVtableClassRegistry()->getClassFromVirtualInstance( extendedMeshShape ) == &hkpStorageExtendedMeshShapeClass )
	{
		const hkpStorageExtendedMeshShape* storageExtendedMeshShape = static_cast<const hkpStorageExtendedMeshShape*>( extendedMeshShape );
		buildShapeDisplayTriangleSubPartsStorage( storageExtendedMeshShape, transform, displayGeometries );
		buildShapeDisplayShapeSubParts( extendedMeshShape, transform, displayGeometries );
	}
	else
	{
		// You can use this alternative approach to building the display geometries from an extended
		// mesh shape. It builds pointer maps to try to avoid sending duplicate vertex data.
		//buildShapeDisplayTriangleSubPartsCompress( extendedMeshShape, transform, displayGeometries );
		//buildShapeDisplayShapeSubParts( extendedMeshShape, transform, displayGeometries );

		// Our default is to convert it to triangles. This creates lots of redundant vertex data but
		// scales better than the compress method.
		buildShapeDisplay_ShapeContainer( extendedMeshShape, transform, displayGeometries );
	}
}

void hkpShapeDisplayBuilder::buildShapeDisplay_BvCompressedMeshShape(const hkpBvCompressedMeshShape* bvCompressedMeshShape, 
																	 const hkTransform& transform, 
																	 hkArray<hkDisplayGeometry*>& displayGeometries,
																	 const hkVector4* scale)
{
	// Continue adding to the current geometry, if there is one.
	hkDisplayGeometry *const displayGeom = getCurrentRawGeometry(displayGeometries);
	hkGeometry *const geom = displayGeom->getGeometry();
	HK_ASSERT(0x4b5bb14e, geom != HK_NULL);
	
	// Obtain and transform mesh geometry and append it to current geometry
	hkGeometry mesh;
	bvCompressedMeshShape->convertToGeometry(mesh);

	// Obtain transform with scale as a 4x4 matrix
	hkMatrix4 transformAsMatrix;
	{	
		hkQsTransform transformWithScale;
		transformWithScale.setFromTransformNoScale(transform);
		if (scale)
		{
			transformWithScale.setScale(*scale);
		}
			
		transformAsMatrix.set( transformWithScale );	
	}

	hkGeometryUtils::transformGeometry(transformAsMatrix, mesh);
	hkGeometryUtils::appendGeometry(mesh, *geom);
}

void hkpShapeDisplayBuilder::buildShapeDisplay_CompressedMeshShape(const hkpCompressedMeshShape* compressedMeshShape,
																   const hkTransform& transform,
																   hkArray<hkDisplayGeometry*>& displayGeometries )
{
	// Continue adding to the current geometry, if there is one.
	hkDisplayGeometry *const displayGeom = getCurrentRawGeometry(displayGeometries);
	hkGeometry *const geom = displayGeom->getGeometry();
	HK_ASSERT(0x4b5bb14e, geom != HK_NULL);

	// Big triangles
	{
		const int numBigTriangles = compressedMeshShape->m_bigTriangles.getSize();
		hkGeometry::Triangle* newTriangle = geom->m_triangles.expandBy( numBigTriangles );
		for ( int i = 0; i < numBigTriangles; ++i )
		{
			hkVector4 v0, v1, v2;
			compressedMeshShape->getBigTriangle( i, v0, v1, v2 );

			v0.setTransformedPos( transform, v0 );
			v1.setTransformedPos( transform, v1 );
			v2.setTransformedPos( transform, v2 );

			const int newIndex = geom->m_vertices.getSize();
			geom->m_vertices.pushBack( v0 );
			geom->m_vertices.pushBack( v1 );
			geom->m_vertices.pushBack( v2 );
			
			newTriangle->m_a = newIndex;
			newTriangle->m_b = newIndex + 1;
			newTriangle->m_c = newIndex + 2;

			++newTriangle;	
		}
	}

	// Chunks
	{
		const int numChunks = compressedMeshShape->m_chunks.getSize();
		for ( int i = 0; i < numChunks; ++i )
		{
			// This chunk may be a transformed version of another.
			const hkpCompressedMeshShape::Chunk& originalChunk = compressedMeshShape->m_chunks[i];
			const int reference = compressedMeshShape->m_chunks[i].m_reference;
			int index = ( reference != 0xffff ) ? reference : i;
			const hkpCompressedMeshShape::Chunk& chunk = compressedMeshShape->m_chunks[index];

			// Vertices
			const int newIndex = geom->m_vertices.getSize();
			const int numVerts = chunk.m_vertices.getSize() / 3;
			hkVector4* newVertex = geom->m_vertices.expandBy( numVerts );

			for ( int j = 0; j < numVerts; ++j )
			{
				const hkUint16* oldVert = chunk.m_vertices.begin() + ( j * 3 );
				hkIntVector iInts; iInts.set( int(oldVert[0]), int(oldVert[1]), int(oldVert[2]), 0 );
				hkVector4 vInts;   iInts.convertS32ToF32(vInts);
				newVertex->setAddMul( chunk.m_offset, vInts, hkSimdReal::fromFloat(compressedMeshShape->m_error) );
				if ( originalChunk.m_transformIndex != 0xffff )
				{
					const hkQsTransform& localTransform = compressedMeshShape->m_transforms[ originalChunk.m_transformIndex ];
					newVertex->setTransformedPos( localTransform, *newVertex );
				}
				newVertex->setTransformedPos( transform, *newVertex );

				++newVertex;
			}

			// Indices
			const int numTriangles = chunk.getNumTriangles();
			hkGeometry::Triangle* newTriangle = geom->m_triangles.expandBy( numTriangles );
			
			int currentIndex = 0;
			{
				// Strips
				const int numStrips = chunk.m_stripLengths.getSize();
				for ( int j = 0; j < numStrips; ++j )
				{
					const int stripLength = chunk.m_stripLengths[j];

					for ( int k = 0; k < stripLength - 2; ++k )
					{
						const int winding = k & 1;
						newTriangle->m_a = newIndex + chunk.m_indices[currentIndex];
						newTriangle->m_b = newIndex + chunk.m_indices[currentIndex + ( 1 + winding )];
						newTriangle->m_c = newIndex + chunk.m_indices[currentIndex + ( 2 - winding )];

						++newTriangle;
						++currentIndex;
					}
					currentIndex += 2;
				}

				// Lists
				while ( currentIndex < chunk.m_indices.getSize() - 2 )
				{
					newTriangle->m_a = newIndex + chunk.m_indices[currentIndex];
					newTriangle->m_b = newIndex + chunk.m_indices[currentIndex + 1];
					newTriangle->m_c = newIndex + chunk.m_indices[currentIndex + 2];

					++newTriangle;
					currentIndex += 3;
				}
			}
		}
	}

	// Convex pieces
	{
		const int numConvexPieces = compressedMeshShape->m_convexPieces.getSize();
		for ( int i = 0; i < numConvexPieces; ++i )
		{
			hkGeometry convexPieceGeom;
			hkpCompressedMeshShapeBuilder::convexPieceToGeometry( compressedMeshShape, i, convexPieceGeom );

			const int newIndex = geom->m_vertices.getSize();

			const int numConvexPieceVertices = convexPieceGeom.m_vertices.getSize();
			hkVector4* vertex = geom->m_vertices.expandBy( numConvexPieceVertices );
			for( int c = 0; c < numConvexPieceVertices; ++c )
			{
				(vertex+c)->setTransformedPos( transform, convexPieceGeom.m_vertices[c] );
			}

			const int numTriangles = convexPieceGeom.m_triangles.getSize();
			hkGeometry::Triangle* newTriangle = geom->m_triangles.expandBy( numTriangles );
			for ( int j = 0; j < numTriangles; ++j )
			{
				newTriangle->m_a = newIndex + convexPieceGeom.m_triangles[j].m_a;
				newTriangle->m_b = newIndex + convexPieceGeom.m_triangles[j].m_b;
				newTriangle->m_c = newIndex + convexPieceGeom.m_triangles[j].m_c;

				++newTriangle;
			}
		}
	}
}


void hkpShapeDisplayBuilder::buildShapeDisplay_SampledHeightField(const hkpSampledHeightFieldShape* heightField,
										  const hkTransform& transform,
										  hkArray<hkDisplayGeometry*>& displayGeometries )
{
	hkDisplayGeometry* displayGeom = getCurrentRawGeometry(displayGeometries);
	hkGeometry* geom = displayGeom->getGeometry();
	HK_ASSERT(0x34673afe, geom != HK_NULL);

	// Convert these vertices to the transformed space.
	hkVector4 scale = heightField->m_intToFloatScale;

	const int startVertex = geom->m_vertices.getSize();
	hkVector4 *const newVertices = geom->m_vertices.expandBy( heightField->m_xRes * heightField->m_zRes );
	hkGeometry::Triangle *const newTriangles = geom->m_triangles.expandBy( ( heightField->m_xRes - 1 ) * ( heightField->m_zRes - 1 ) * 2 );

	for ( int i = 0; i < heightField->m_xRes; ++i )
	{
		for ( int j = 0; j < heightField->m_zRes; ++j )
		{
			hkVector4 p00; 
			p00.set( (hkReal) i, heightField->getHeightAt( i, j ), (hkReal) j ); 
			p00.mul( scale );
			newVertices[ ( i * heightField->m_zRes ) + j ].setTransformedPos(transform, p00 );
		}
	}
	for ( int i = 0; i < heightField->m_xRes - 1; ++i )
	{
		for ( int j = 0; j < heightField->m_zRes - 1; ++j )
		{
			const int thisRowV = startVertex + i * heightField->m_zRes;
			const int nextRowV = thisRowV + heightField->m_zRes;
			const int thisRowI = i * ( heightField->m_zRes - 1 ) * 2;
			if ( heightField->getTriangleFlip())
			{
				newTriangles[ thisRowI + ( j * 2 ) ].set( thisRowV + j, thisRowV + j + 1, nextRowV + j + 1 );
				newTriangles[ thisRowI + ( j * 2 ) + 1 ].set( thisRowV + j, nextRowV + j + 1, nextRowV + j );
			}
			else
			{
				newTriangles[ thisRowI + ( j * 2 ) ].set( thisRowV + j, thisRowV + j + 1, nextRowV + j );
				newTriangles[ thisRowI + ( j * 2 ) + 1 ].set( thisRowV + j + 1, nextRowV + j + 1, nextRowV + j );
			}
		}
	}
}


void hkpShapeDisplayBuilder::buildShapeDisplay_ShapeUnregistered(const hkpShape* shape, const hkTransform& transform,
																 hkArray<hkDisplayGeometry*>& displayGeometries)
{
	// If the shape type is not registered use a hollow box around its current aabb as display geometry
	HK_REPORT("Shape type unsupported. Using current aabb for display geometry");
	hkAabb aabb; 
	shape->getAabb(hkTransform::getIdentity(), 0, aabb);

	// Find out dimensions and calculate edge thickness
	hkVector4 dim; dim.setSub(aabb.m_max, aabb.m_min);
	hkVector4 halfDim; halfDim.setMul(dim, hkSimdReal::getConstant<HK_QUADREAL_INV_2>());
	hkSimdReal halfThickness = halfDim.horizontalMin<3>() * hkSimdReal::getConstant<HK_QUADREAL_INV_6>();	
	hkVector4 halfExtents;

	// Create edge along X axis and place it in the four positions
	halfExtents.set(halfDim.getComponent<0>() + halfThickness, halfThickness, halfThickness, halfThickness);	
	hkSimdReal zero; zero.setZero();
	for (int pos = 0; pos < 4; ++pos)
	{
		hkDisplayBox* displayBox = new hkDisplayBox(halfExtents);
		hkVector4 translation; 
		translation.set(halfDim.getComponent<0>(), 
						(pos & 1) ? dim.getComponent<1>() : zero, 	
						(pos & 2) ? dim.getComponent<2>() : zero,
						zero);
		translation.add(aabb.m_min);
		hkTransform translationOnly;
		translationOnly.setIdentity();
		translationOnly.setTranslation(translation);
		displayBox->setTransform(translationOnly);	
		displayGeometries.pushBack(displayBox);	
	}		

	// Create edge along Y axis and place it in the four positions
	halfExtents.set(halfThickness, halfDim.getComponent<1>() + halfThickness, halfThickness, halfThickness);	
	for (int pos = 0; pos < 4; ++pos)
	{
		hkDisplayBox* displayBox = new hkDisplayBox(halfExtents);
		hkVector4 translation; 
		translation.set((pos & 1) ? dim.getComponent<0>() : zero, 
						halfDim.getComponent<1>(), 
						(pos & 2) ? dim.getComponent<2>() : zero,
						zero);
		translation.add(aabb.m_min);
		hkTransform translationOnly;
		translationOnly.setIdentity();
		translationOnly.setTranslation(translation);
		displayBox->setTransform(translationOnly);	
		displayGeometries.pushBack(displayBox);	
	}

	// Create edge along Z axis and place it in the four positions
	halfExtents.set(halfThickness, halfThickness, halfDim.getComponent<2>() + halfThickness, halfThickness);	
	for (int pos = 0; pos < 4; ++pos)
	{
		hkDisplayBox* displayBox = new hkDisplayBox(halfExtents);
		hkVector4 translation; 
		translation.set((pos & 1) ? dim.getComponent<0>() : zero, 
						(pos & 2) ? dim.getComponent<1>() : zero,
						halfDim.getComponent<2>(),
						zero);
		translation.add(aabb.m_min);
		hkTransform translationOnly;
		translationOnly.setIdentity();
		translationOnly.setTranslation(translation);
		displayBox->setTransform(translationOnly);	
		displayGeometries.pushBack(displayBox);	
	}
}


void hkpShapeDisplayBuilder::buildShapeDisplay_ShapeContainer(const hkpShapeContainer* shapeContainer, 
									  const hkTransform& transform,
									  hkArray<hkDisplayGeometry*>& displayGeometries )
{
	hkpShapeBuffer buffer;

	for (hkpShapeKey key = shapeContainer->getFirstKey(); key != HK_INVALID_SHAPE_KEY; key = shapeContainer->getNextKey( key ) )
	{
		const hkpShape* child = shapeContainer->getChildShape( key, buffer );
		buildShapeDisplay( child, transform, displayGeometries );
	}
}


void hkpShapeDisplayBuilder::buildShapeDisplay_Sphere( const hkpSphereShape* sphereShape, const hkTransform& transform, 
													   hkArray<hkDisplayGeometry*>& displayGeometries,
													   const hkVector4* scale )
{
	hkReal radius = sphereShape->getRadius();

	if (scale)
	{
		HK_ON_DEBUG( hkVector4 scaleX; scaleX.setAll(scale->getComponent<0>()); )
		HK_WARN_ON_DEBUG_IF(!scale->allExactlyEqual<3>(scaleX), 0x39d9a3a3, "Shape type not supported with non-uniform scale");
		radius *= hkMath::abs((*scale)(0));
	}
		
	hkSphere sphere(hkVector4::getZero(), radius);
	hkDisplaySphere* displaySphere = new hkDisplaySphere(sphere, m_environment.m_sphereThetaRes, m_environment.m_spherePhiRes);
	displaySphere->setTransform( transform );
	displayGeometries.pushBack(displaySphere);
}

void hkpShapeDisplayBuilder::buildShapeDisplay_MultiSphere( const hkpMultiSphereShape* s, const hkTransform& transform, hkArray<hkDisplayGeometry*>& displayGeometries )
{
	const hkVector4* v = s->getSpheres();
	for( int i = 0; i < s->getNumSpheres(); ++i )
	{
		hkSphere sphere( hkVector4::getZero(), v[i](3) );
		hkDisplaySphere* displaySphere = new hkDisplaySphere( sphere, m_environment.m_sphereThetaRes, m_environment.m_spherePhiRes );

		displaySphere->setTranslation( v[i] );
		displayGeometries.pushBack(displaySphere);
	}
}

void hkpShapeDisplayBuilder::buildShapeDisplay_Plane( const hkpPlaneShape* planeShape, const hkTransform& transform, hkArray<hkDisplayGeometry*>& displayGeometries )
{
	const hkVector4& plane = planeShape->getPlane();
	const hkVector4& aabbCenter = planeShape->getAabbCenter();
	const hkVector4& aabbHalfExtents = planeShape->getAabbHalfExtents();

	const int component = plane.getIndexOfMaxAbsComponent<3>();

	// Pick four corners from a face of the aabb and project them onto the plane along the component axis.
	hkArray<hkVector4> points;
	{
		hkSimdReal div; div.setReciprocal<HK_ACC_FULL,HK_DIV_IGNORE>(plane.getComponent(component));
		// These are chosen so the points arise in sequence.
		static const hkReal signsForNonComponentAxes[4][2] = { { -1.0f, -1.0f }, { -1.0f, 1.0f }, { 1.0f, 1.0f }, { 1.0f, -1.0f } };
		for ( int i = 0; i < 4; ++i )
		{
			hkVector4& point = points.expandOne();

			HK_ALIGN_REAL(hkReal signs[3]);
			{
				signs[component] = 1.0f;
				signs[(component + 1) % 3] = signsForNonComponentAxes[i][0];
				signs[(component + 2) % 3] = signsForNonComponentAxes[i][1];
			}
			point.load<3>( &signs[0] );
			point.mul( aabbHalfExtents );
			point.add( aabbCenter );

			// Project the point onto the plane along the component axis.
			const hkSimdReal t = -div * plane.dot4xyz1( point );
			point.addMul(hkVector4::getConstant((hkVectorConstant)(HK_QUADREAL_1000 + component)), t);
		}
	}

	// We need to crop the rectangle to the AABB. We do this by walking
	// through the list of points, adding new ones and removing old ones.
	// We need only consider the planes perpendicular to the component axis.
	{
		// We need to duplicate the first point, in case we remove it.
		hkVector4 firstPoint = points[0];
		points.pushBack( firstPoint );
		hkSimdReal faceT[2];
		{
			faceT[0] = aabbCenter.getComponent( component ) - aabbHalfExtents.getComponent( component );
			faceT[1] = aabbCenter.getComponent( component ) + aabbHalfExtents.getComponent( component );
		}
		int i = 0;
		do
		{
			hkVector4& startPoint = points[i];
			hkVector4& endPoint = points[i + 1];
			const hkSimdReal startPointT = startPoint.getComponent( component );
			const hkSimdReal endPointT = endPoint.getComponent( component );
			// Find the first plane this edge crosses, if there is one.
			int face = -1;
			hkSimdReal t = hkSimdReal_Max;
			for ( int j = 0; j < 2; ++j )
			{
				if ( ( ( startPointT < faceT[j] ) && ( faceT[j] < endPointT ) ) || ( ( endPointT < faceT[j] ) && ( faceT[j] < startPointT  ) ) )
				{
					const hkSimdReal newT = ( faceT[j] - startPointT ) / ( endPointT - startPointT );
					if ( newT < t )
					{
						t = newT;
						face = j;
					}
				}
			}
			// Did we cross a face?
			if ( face != -1 )
			{
				hkVector4 newPoint;
				newPoint.setInterpolate( startPoint, endPoint, t );
				// Correct for numerical problems.
				newPoint.setComponent( component, faceT[face] );
				points.insertAt( i + 1, newPoint );
			}
			// Is the current start point outside the AABB?
			if ( ( startPointT < faceT[0] ) || ( startPointT > faceT[1] ) )
			{
				// Yes: remove it.
				points.removeAtAndCopy( i );
			}
			else
			{
				// No: keep it.
				++i;
			}
		} while ( i < points.getSize() - 1 ); // -1 because of the extra first point.
		// Remove the extra first point.
		points.popBack();
	}

	// Create a fan of triangles from the points.
	{
		hkDisplayGeometry* displayGeom = getCurrentRawGeometry(displayGeometries);
		hkGeometry* geom = displayGeom->getGeometry();
		HK_ASSERT(0x34673afe, geom != HK_NULL);

		const int numPoints = points.getSize();

		const int startVertex = geom->m_vertices.getSize();
		hkVector4 *const newVertices = geom->m_vertices.expandBy( numPoints );
		hkGeometry::Triangle *const newTriangles = geom->m_triangles.expandBy( numPoints - 2 );

		const int triangleWinding = plane.getComponent( component ).isGreaterZero() ? 1 : 0;

		for ( int i = 0; i < numPoints; ++i )
		{
			newVertices[i].setTransformedPos( transform, points[i] );
		}
		for ( int i = 0; i < numPoints - 2; ++i )
		{
			newTriangles[i].m_a = startVertex;
			newTriangles[i].m_b = startVertex + i + ( 1 + triangleWinding );
			newTriangles[i].m_c = startVertex + i + ( 2 - triangleWinding );
		}
	}
}

void hkpShapeDisplayBuilder::buildShapeDisplay_Capsule( const hkpCapsuleShape* s, const hkTransform& transform, 
													    hkArray<hkDisplayGeometry*>& displayGeometries,
													    const hkVector4* scale )
{
	hkVector4 vertexA = s->getVertex<0>();
	hkVector4 vertexB = s->getVertex<1>();
	hkReal radius = s->getRadius();

	// Apply scale if present
	if (scale)
	{
		const hkSimdReal scale0 = scale->getComponent<0>();
		hkVector4 scaleX; scaleX.setAll(scale0);
		HK_WARN_ON_DEBUG_IF(!scale->allExactlyEqual<3>(scaleX), 0x39d9a3a3, "Shape type not supported with non-uniform scale");
		vertexA.mul(scaleX);
		vertexB.mul(scaleX);
		hkSimdReal absScale0; absScale0.setAbs(scale0);
		radius *= absScale0.getReal();
	}

	hkDisplayCapsule* displayCapsule = new hkDisplayCapsule( vertexA, vertexB, radius );
	displayCapsule->setTransform( transform );
	displayGeometries.pushBack( displayCapsule );
}

void hkpShapeDisplayBuilder::buildShapeDisplay_Cylinder( const hkpCylinderShape* s, const hkTransform& transform, 
														 hkArray<hkDisplayGeometry*>& displayGeometries,
														 const hkVector4* scale)
{
	hkVector4 vertexA = s->getVertex<0>();
	hkVector4 vertexB = s->getVertex<1>();
	hkReal radius = s->getCylinderRadius();

	// Apply scale if present
	if (scale)
	{
		const hkSimdReal scale0 = scale->getComponent<0>();
		hkVector4 scaleX; scaleX.setAll(scale0);
		HK_WARN_ON_DEBUG_IF(!scale->allExactlyEqual<3>(scaleX), 0x39d9a3a3, "Shape type not supported with non-uniform scale");
		vertexA.mul(scaleX);
		vertexB.mul(scaleX);
		hkSimdReal absScale0; absScale0.setAbs(scale0);
		radius *= absScale0.getReal();
	}

	hkDisplayCylinder* displayCylinder = new hkDisplayCylinder( vertexA, vertexB, radius );
	displayCylinder->setTransform( transform );
	displayGeometries.pushBack( displayCylinder );
}

void hkpShapeDisplayBuilder::buildShapeDisplay_MultiRay( const hkpMultiRayShape* s, const hkTransform& transform, hkArray<hkDisplayGeometry*>& displayGeometries )
{
	hkDisplayGeometry* displayGeom = getCurrentRawGeometry(displayGeometries);
	hkGeometry* geom = displayGeom->getGeometry();
	HK_ASSERT(0x142cb874, geom != HK_NULL);

	int vertBase = 0;

	for(int j = 0; j < s->getRays().getSize(); j++)
	{
		hkpMultiRayShape::Ray seg = s->getRays()[j];

		hkVector4& start = *geom->m_vertices.expandBy(1);
		start = seg.m_start;
		start.setTransformedPos( transform, start );

		hkVector4& joggle = *geom->m_vertices.expandBy(1);
		joggle = seg.m_start;
		hkVector4 offset; offset.set( 0.01f, 0, 0 );
		joggle.add( offset );

		hkVector4& end = *geom->m_vertices.expandBy(1);
		end = seg.m_end;
		end.setTransformedPos( transform, end );

		hkGeometry::Triangle& tri = *geom->m_triangles.expandBy(1);
		tri.set(vertBase, vertBase + 1, vertBase + 2);

		vertBase += 3;
	}
}

void hkpShapeDisplayBuilder::buildShapeDisplay_Box( const hkpBoxShape* boxShape, const hkTransform& transform, 
												    hkArray<hkDisplayGeometry*>& displayGeometries,
													const hkVector4* scale )
{
	hkVector4 halfExtents = boxShape->getHalfExtents();
	if (scale)
	{
		halfExtents.mul(*scale);
		halfExtents.setAbs(halfExtents);
	}
	hkDisplayBox* displayBox = new hkDisplayBox(halfExtents);
	displayBox->setTransform(transform);
	displayGeometries.pushBack(displayBox);
}

void hkpShapeDisplayBuilder::buildShapeDisplay_ListShape( const hkpListShape* listShape, const hkTransform& transform, hkArray<hkDisplayGeometry*>& displayGeometries )
{
	for ( hkpShapeKey key = listShape->getFirstKey(); key != HK_INVALID_SHAPE_KEY; key = listShape->getNextKey( key ) )
	{
		const hkpShape* child = listShape->getChildShapeInl( key );

		buildShapeDisplay( child, transform, displayGeometries );
	}
}

void hkpShapeDisplayBuilder::buildShapeDisplay_Triangle( const hkpTriangleShape* triangleShape, const hkTransform& transform, hkArray<hkDisplayGeometry*>& displayGeometries, const hkVector4* scale )
{
	hkDisplayGeometry* displayGeom = getCurrentRawGeometry(displayGeometries);
	hkGeometry* geom = displayGeom->getGeometry();
	HK_ASSERT(0x4b5bb14e, geom != HK_NULL);

	hkQsTransform transformWithScale; transformWithScale.setFromTransformNoScale(transform);
	if (scale)
	{
		const hkSimdReal scale0 = scale->getComponent<0>();
		hkVector4 scaleX; scaleX.setAll(scale0);
		HK_WARN_ON_DEBUG_IF(!scale->allExactlyEqual<3>(scaleX), 0x39d9a3a3, "Shape type not supported with non-uniform scale");
		transformWithScale.setScale(*scale);
	}

	// Convert these vertices to the transformed space.
	int vertBase = geom->m_vertices.getSize();

	hkVector4 *const vertices = geom->m_vertices.expandBy( 3 );
	vertices[0].setTransformedPos(transformWithScale, triangleShape->getVertex<0>());
	vertices[1].setTransformedPos(transformWithScale, triangleShape->getVertex<1>());
	vertices[2].setTransformedPos(transformWithScale, triangleShape->getVertex<2>());

	hkGeometry::Triangle& tri = *geom->m_triangles.expandBy(1);
	tri.set(vertBase, vertBase + 1, vertBase + 2);

	if ( triangleShape->getExtrusion().lengthSquared<3>().isGreaterZero() )
	{
		hkVector4 ext0; ext0.setAdd(triangleShape->getVertex<0>(), triangleShape->getExtrusion());
		hkVector4 ext1; ext1.setAdd(triangleShape->getVertex<1>(), triangleShape->getExtrusion());
		hkVector4 ext2; ext2.setAdd(triangleShape->getVertex<2>(), triangleShape->getExtrusion());

		hkVector4 *const extrudedVertices = geom->m_vertices.expandBy( 3 );
		extrudedVertices[0].setTransformedPos(transformWithScale, ext0);
		extrudedVertices[1].setTransformedPos(transformWithScale, ext1);
		extrudedVertices[2].setTransformedPos(transformWithScale, ext2);

		hkGeometry::Triangle* extrudedTris = geom->m_triangles.expandBy(7);
		extrudedTris[0].set(vertBase + 3, vertBase + 1, vertBase + 0);
		extrudedTris[1].set(vertBase + 3, vertBase + 4, vertBase + 1);
		extrudedTris[2].set(vertBase + 4, vertBase + 5, vertBase + 1);
		extrudedTris[3].set(vertBase + 5, vertBase + 2, vertBase + 1);
		extrudedTris[4].set(vertBase + 5, vertBase + 0, vertBase + 2);
		extrudedTris[5].set(vertBase + 5, vertBase + 3, vertBase + 0);
		extrudedTris[6].set(vertBase + 5, vertBase + 4, vertBase + 3);
	}
}

void hkpShapeDisplayBuilder::buildShapeDisplay_ConvexVertices( const hkpConvexVerticesShape* cvShape, 
															   const hkTransform& transform, 
															   hkArray<hkDisplayGeometry*>& displayGeometries,
															   const hkVector4* scale )
{
	// Obtain collision spheres
	const int numSpheres = cvShape->getNumCollisionSpheres();
	if( numSpheres == 0 )
	{
		HK_WARN(0x1e70ea5e, "Not making a display shape for a convex vertices shape with no vertices");
		return;
	}
	hkLocalArray<hkSphere> vertices(numSpheres); vertices.setSize( numSpheres );
	const hkSphere* spheres = cvShape->getCollisionSpheres(vertices.begin());

	// Compute transform with scale
	hkQsTransform transformWithScale; transformWithScale.setFromTransformNoScale(transform);
	if (scale)
	{
		transformWithScale.setScale(*scale);
	}

	// Transform spheres
	hkArray<hkVector4> transformedVertices;
	transformedVertices.setSize( numSpheres );	
	for(int i = 0; i < numSpheres; i++)
	{
		transformedVertices[i].setTransformedPos(transformWithScale, spheres[i].getPosition());
	}
	
	// HVK-5032
	hkGeometry* outputGeom = new hkGeometry;
	hkGeometryUtility::createConvexGeometry(transformedVertices,*outputGeom);
	hkDisplayConvex* displayGeom = new hkDisplayConvex(outputGeom);
	displayGeometries.pushBack(displayGeom);
}

void hkpShapeDisplayBuilder::buildShapeDisplay_ConvexPiece( const hkpConvexPieceShape* triangulatedConvexShape, const hkTransform& transform, hkArray<hkDisplayGeometry*>& displayGeometries )
{
	// Create the geometry
	hkGeometry* outputGeom = new hkGeometry;

	hkpShapeBuffer buffer2;

	for ( int i = 0 ; i < triangulatedConvexShape->m_numDisplayShapeKeys ; i++ )
	{
		const hkpTriangleShape& triangleShape = *( static_cast< const hkpTriangleShape* >( 
			triangulatedConvexShape->m_displayMesh->getChildShape( triangulatedConvexShape->m_displayShapeKeys[i], buffer2 ) ));

		// pushback information about this triangle to the new geometry.
		hkGeometry::Triangle& tri = *outputGeom->m_triangles.expandBy(1);

		int vertexSize = outputGeom->m_vertices.getSize();
		tri.set( vertexSize, vertexSize+1, vertexSize+2	);

		for ( int j = 0 ; j < 3 ; j++ )
		{
			hkVector4& transformedVertex = *outputGeom->m_vertices.expandBy(1);
			transformedVertex.setTransformedPos(transform, triangleShape.getVertex( j ));
		}
	}

	hkDisplayConvex* displayGeom = new hkDisplayConvex(outputGeom);
	displayGeometries.pushBack(displayGeom);
}

hkBool hkpShapeDisplayBuilder::buildShapeDisplay_UserShapes( const hkpShape* shape, const hkTransform& transform, hkArray<hkDisplayGeometry*>& displayGeometries )
{
	hkBool handled = false;
	for (int i = 0; i < hkpUserShapeDisplayBuilder::getInstance().m_userShapeBuilders.getSize(); ++i )
	{
		if ( hkpUserShapeDisplayBuilder::getInstance().m_userShapeBuilders[i].type == shape->getType() )
		{
			hkpUserShapeDisplayBuilder::getInstance().m_userShapeBuilders[i].f( shape, transform, displayGeometries, this );
			handled = true;
			continue;
		}
	}
	return handled;
}

void hkpShapeDisplayBuilder::buildShapeDisplay_StaticCompound( const hkpStaticCompoundShape* staticCompoundShape, 
															   const hkTransform& transform, 
															   hkArray<hkDisplayGeometry*>& displayGeometries,
															   const hkVector4* scale )
{
	hkQsTransform transformWithScale; transformWithScale.setFromTransformNoScale(transform);
	if (scale)
	{		
		transformWithScale.setScale(*scale);
	}

	const hkArray<hkpStaticCompoundShape::Instance>& instances = staticCompoundShape->getInstances();
	for (int i = 0; i < instances.getSize(); i++)
	{		
		hkQsTransform childTransform; childTransform.setMulScaled(transformWithScale, instances[i].getTransform());
		hkTransform childTransformNoScale; childTransform.copyToTransformNoScale(childTransformNoScale);
		
		// Test if the child transform has identity scaling
		if( childTransform.m_scale.allEqual<3>( hkVector4::getConstant<HK_QUADREAL_1>(), hkSimdReal_Eps ) )
		{			
			buildShapeDisplay( instances[i].getShape(), childTransformNoScale, displayGeometries );
		}
		else
		{			
			buildShapeDisplay( instances[i].getShape(), childTransformNoScale, displayGeometries, &childTransform.m_scale );
		}
	}
}


// This is the alternative to having a buildDisplayGeometry as a virtual function in Shape.
void hkpShapeDisplayBuilder::buildShapeDisplay( const hkpShape* shape, const hkTransform& transform, 
											    hkArray<hkDisplayGeometry*>& displayGeometries, const hkVector4* scale )
{
	// Try first with user shape types to avoid overriding them with the hardcoded shape types
	if (buildShapeDisplay_UserShapes(shape, transform, displayGeometries))
	{
		return;
	}

	switch ( shape->getType() )
	{
		case hkcdShapeType::SPHERE:
		{
			const hkpSphereShape* sphereShape = static_cast<const hkpSphereShape*>(shape);
			buildShapeDisplay_Sphere( sphereShape, transform, displayGeometries, scale );
			break;
		}
		case hkcdShapeType::MULTI_SPHERE:
		{
			HK_WARN_ON_DEBUG_IF(scale, 0x5ba9c97f, "Scale is not supported for this shape type and will be ignored");			
			const hkpMultiSphereShape* s = static_cast<const hkpMultiSphereShape*>(shape);
			buildShapeDisplay_MultiSphere( s, transform, displayGeometries );
			break;
		}
		case hkcdShapeType::PLANE:
		{
			HK_WARN_ON_DEBUG_IF(scale, 0x5ba9c97f, "Scale is not supported for this shape type and will be ignored");			
			const hkpPlaneShape* planeShape = static_cast<const hkpPlaneShape*>(shape);
			buildShapeDisplay_Plane( planeShape, transform, displayGeometries );
			break;
		}
		case hkcdShapeType::CAPSULE:
		{
			const hkpCapsuleShape* s = static_cast<const hkpCapsuleShape*>(shape);
			buildShapeDisplay_Capsule( s, transform, displayGeometries, scale );
			break;
		}
		case hkcdShapeType::CYLINDER:
		{
			const hkpCylinderShape* s = static_cast<const hkpCylinderShape*>(shape);
			buildShapeDisplay_Cylinder( s, transform, displayGeometries, scale );
			break;
		}

		case hkcdShapeType::MULTI_RAY:
		{
			// TODO
			HK_WARN_ON_DEBUG_IF(scale, 0x5ba9c97f, "Scale is not supported for this shape type and will be ignored");			
			const hkpMultiRayShape* s = static_cast<const hkpMultiRayShape*>(shape);
			buildShapeDisplay_MultiRay( s, transform, displayGeometries );
			break;
		}
		case hkcdShapeType::BOX:
		{
			const hkpBoxShape* boxShape = static_cast<const hkpBoxShape*>(shape);
			buildShapeDisplay_Box( boxShape, transform, displayGeometries, scale );
			break;
		}
		
		case hkcdShapeType::CONVEX_TRANSLATE:
		{			
			const hkpConvexTranslateShape* convexTranslate = static_cast<const hkpConvexTranslateShape*>(shape);			

			// Obtain translation
			hkVector4 translation = convexTranslate->getTranslation();
			if (scale)
			{
				translation.mul(*scale);				
			}

			// Compute child transform
			hkTransform childTransform; 
			{			
				hkTransform translateTransform; translateTransform.setIdentity();	
				translateTransform.setTranslation(translation);
				childTransform.setMul(transform, translateTransform);
			}

			buildShapeDisplay(convexTranslate->getChildShape(), childTransform, displayGeometries, scale);

			break;
		}

		case hkcdShapeType::CONVEX_TRANSFORM:
		{
			HK_WARN_ON_DEBUG_IF(scale, 0x5ba9c97f, "Scale is not supported for this shape type and will be ignored");			
			const hkpConvexTransformShape* convexTransformShape = static_cast<const hkpConvexTransformShape*>(shape);
			
			// Calculate child transform without scale
			const hkQsTransform& convexTransform = convexTransformShape->getQsTransform();			
			const hkVector4& childScale = convexTransform.getScale();
			hkTransform convexTransformNoScale; convexTransform.copyToTransformNoScale(convexTransformNoScale);
			hkTransform childTransform; childTransform.setMul(transform, convexTransformNoScale);			

			buildShapeDisplay(convexTransformShape->getChildShape(), childTransform, displayGeometries, &childScale);						
			break;
		}	

		case hkcdShapeType::TRANSFORM:
		{
			HK_WARN_ON_DEBUG_IF(scale, 0x5ba9c97f, "Scale is not supported for this shape type and will be ignored");			
			const hkpTransformShape* ts = static_cast<const hkpTransformShape*>( shape );
			hkTransform T; T.setMul( transform, ts->getTransform() );

			buildShapeDisplay( ts->getChildShape(), T, displayGeometries);

			break;
		}
		case hkcdShapeType::BV:
		{
			HK_WARN_ON_DEBUG_IF(scale, 0x5ba9c97f, "Scale is not supported for this shape type and will be ignored");			
			const hkpBvShape* bvShape = static_cast<const hkpBvShape*>(shape);
			buildShapeDisplay( bvShape->getBoundingVolumeShape(), transform, displayGeometries);
			break;
		}
		/*
		case HK_SHAPE_CONVEX_WELDER:
		{
			const hkConvexWelderShape* cxWeldShape = static_cast<const hkConvexWelderShape*>(shape);
			shape = cxWeldShape->m_compoundShapeToBeWelded;
		}
		*/

		case hkcdShapeType::MOPP:
		{
			HK_WARN_ON_DEBUG_IF(scale, 0x5ba9c97f, "Scale is not supported for this shape type and will be ignored");			
			const hkpMoppBvTreeShape* bvShape = static_cast<const hkpMoppBvTreeShape*>(shape);
			buildShapeDisplay( bvShape->getChild(), transform, displayGeometries );
			break;
		}

		case hkcdShapeType::LIST:
		{
			HK_WARN_ON_DEBUG_IF(scale, 0x5ba9c97f, "Scale is not supported for this shape type and will be ignored");			
			const hkpListShape* listShape = static_cast<const hkpListShape*>(shape);
			buildShapeDisplay_ListShape( listShape, transform, displayGeometries );
			break;
		}
		case hkcdShapeType::BV_TREE:
		case hkcdShapeType::CONVEX_LIST:
		case hkcdShapeType::COLLECTION:
		case hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_COLLECTION:
		case hkcdShapeType::TRIANGLE_COLLECTION:
		case hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_BV_TREE:
		{
			HK_WARN_ON_DEBUG_IF(scale, 0x5ba9c97f, "Scale is not supported for this shape type and will be ignored");			
			const hkpShapeContainer* shapeContainer = shape->getContainer();
			
			buildShapeDisplay_ShapeContainer( shapeContainer, transform, displayGeometries );
		
			break;
		}
		case hkcdShapeType::EXTENDED_MESH:
		{
			HK_WARN_ON_DEBUG_IF(scale, 0x5ba9c97f, "Scale is not supported for this shape type and will be ignored");			
			const hkpExtendedMeshShape *const extendedMeshShape = static_cast<const hkpExtendedMeshShape*>( shape );

			buildShapeDisplay_ExtendedMeshShape( extendedMeshShape, transform, displayGeometries );

 			break;
		}
		case hkcdShapeType::COMPRESSED_MESH:
		{
			HK_WARN_ON_DEBUG_IF(scale, 0x5ba9c97f, "Scale is not supported for this shape type and will be ignored");			
			const hkpCompressedMeshShape *const compressedMeshShape = static_cast<const hkpCompressedMeshShape*>( shape );

			buildShapeDisplay_CompressedMeshShape( compressedMeshShape, transform, displayGeometries );

			break;
		}
		case hkcdShapeType::BV_COMPRESSED_MESH:
		{			
			const hkpBvCompressedMeshShape* const bvCompressedMeshShape = static_cast<const hkpBvCompressedMeshShape*>(shape);			
			buildShapeDisplay_BvCompressedMeshShape(bvCompressedMeshShape, transform, displayGeometries, scale);
			break;
		}
		case hkcdShapeType::TRIANGLE:
		{		
			const hkpTriangleShape* triangleShape = static_cast<const hkpTriangleShape*>(shape);
			buildShapeDisplay_Triangle( triangleShape, transform, displayGeometries, scale );
			break;
		}

		case hkcdShapeType::CONVEX_VERTICES:
		{
			const hkpConvexVerticesShape* cvShape = static_cast<const hkpConvexVerticesShape*>(shape);			
			buildShapeDisplay_ConvexVertices( cvShape, transform, displayGeometries, scale );
			break;
		}

		case hkcdShapeType::CONVEX_PIECE:
		{
			HK_WARN_ON_DEBUG_IF(scale, 0x5ba9c97f, "Scale is not supported for this shape type and will be ignored");			
			const hkpConvexPieceShape* triangulatedConvexShape = static_cast<const hkpConvexPieceShape*>(shape);
			buildShapeDisplay_ConvexPiece( triangulatedConvexShape, transform, displayGeometries );
			break;
		}

		case hkcdShapeType::SAMPLED_HEIGHT_FIELD:
		{
			HK_WARN_ON_DEBUG_IF(scale, 0x5ba9c97f, "Scale is not supported for this shape type and will be ignored");			
			const hkpSampledHeightFieldShape* heightField = static_cast<const hkpSampledHeightFieldShape*>(shape);

			buildShapeDisplay_SampledHeightField( heightField, transform, displayGeometries );

			break;
		}		

		case hkcdShapeType::STATIC_COMPOUND:
		{			
			const hkpStaticCompoundShape* staticCompoundShape = static_cast<const hkpStaticCompoundShape*>(shape);
			buildShapeDisplay_StaticCompound(staticCompoundShape, transform, displayGeometries, scale);
			break;
		}

		default:
			HK_WARN_ON_DEBUG_IF(scale, 0x5ba9c97f, "Scale is not supported for this shape type and will be ignored");
			buildShapeDisplay_ShapeUnregistered(shape, transform, displayGeometries);
		
	}
}

hkBool hkpShapeDisplayBuilder::buildPartialShapeDisplay_MultiSphere( const hkpMultiSphereShape* s, const hkTransform& transform, int branchDepth, int& numSimpleShapes, hkpShapeContinueData* continueData, hkArray<hkDisplayGeometry*>& displayGeometries )
{
	HK_ASSERT2( 0x3e421e73, numSimpleShapes > 0, "Cannot build a shape display for 0 numSimpleShapes." );
	HK_ASSERT2( 0x3e421e82, branchDepth == continueData->m_shapeKeys.getSize(), "The continue data is inconsistent with the traversal of the shape hierarchy." );
	
	if ( continueData->m_i == -1 )
	{
		// This is the first time we've processed this shape:
		continueData->m_i = 0;
	}
	
	const hkVector4* v = s->getSpheres();
	while ( ( continueData->m_i < s->getNumSpheres() ) && ( numSimpleShapes > 0 ) )
	{
		hkSphere sphere( hkVector4::getZero(), v[continueData->m_i](3) );
		hkDisplaySphere* displaySphere = new hkDisplaySphere( sphere, m_environment.m_sphereThetaRes, m_environment.m_spherePhiRes );

		displaySphere->setTranslation( v[continueData->m_i] );
		displayGeometries.pushBack(displaySphere);
		++continueData->m_i;
		--numSimpleShapes;
	}

	if ( continueData->m_i == s->getNumSpheres() )
	{
		// We finished processing this subshape.
		continueData->m_i = -1;
		return true;
	}
	else
	{
		return false;
	}
}

hkBool hkpShapeDisplayBuilder::buildPartialShapeDisplay_MultiRay( const hkpMultiRayShape* s, const hkTransform& transform, int branchDepth, int& numSimpleShapes, hkpShapeContinueData* continueData, hkArray<hkDisplayGeometry*>& displayGeometries )
{
	
	HK_ASSERT2( 0x3e421e73, numSimpleShapes > 0, "Cannot build a shape display for 0 numSimpleShapes." );
	HK_ASSERT2( 0x3e421e82, branchDepth == continueData->m_shapeKeys.getSize(), "The continue data is inconsistent with the traversal of the shape hierarchy." );
	
	hkDisplayGeometry* displayGeom = getCurrentRawGeometry(displayGeometries);
	hkGeometry* geom = displayGeom->getGeometry();
	HK_ASSERT(0x142cb874, geom != HK_NULL);

	if ( continueData->m_i == -1 )
	{
		// This is the first time we've processed this shape:
		continueData->m_i = 0;
	}

	int vertBase = 0;

	while ( ( continueData->m_i < s->getRays().getSize() ) && ( numSimpleShapes > 0 ) )
	{
		hkpMultiRayShape::Ray seg = s->getRays()[continueData->m_i];

		hkVector4& start = *geom->m_vertices.expandBy(1);
		start = seg.m_start;
		start.setTransformedPos( transform, start );

		hkVector4& joggle = *geom->m_vertices.expandBy(1);
		joggle = seg.m_start;
		hkVector4 offset; offset.set( 0.01f, 0, 0 );
		joggle.add( offset );

		hkVector4& end = *geom->m_vertices.expandBy(1);
		end = seg.m_end;
		end.setTransformedPos( transform, end );

		hkGeometry::Triangle& tri = *geom->m_triangles.expandBy(1);
		tri.set(vertBase, vertBase + 1, vertBase + 2);

		vertBase += 3;

		++continueData->m_i;
		--numSimpleShapes;
	}

	if ( continueData->m_i == s->getRays().getSize() )
	{
		// We finished processing this shape.
		continueData->m_i = -1;
		return true;
	}
	else
	{
		return false;
	}
}

hkBool hkpShapeDisplayBuilder::buildPartialShapeDisplay_ShapeContainer( const hkpShapeContainer* shapeContainer, 
																		const hkTransform& transform, 
																	    int branchDepth, int& numSimpleShapes, 
																		hkpShapeContinueData* continueData, 
																		hkArray<hkDisplayGeometry*>& displayGeometries,
																		const hkVector4* scale )
{
	HK_ASSERT2( 0x3e421e73, numSimpleShapes > 0, "Cannot build a shape display for 0 numSimpleShapes." );
	const int totalDepthSoFar = continueData->m_shapeKeys.getSize();
	HK_ASSERT2( 0x3e421e82, branchDepth <= totalDepthSoFar, "The continue data is inconsistent with the traversal of the shape hierarchy." );
	if ( totalDepthSoFar == branchDepth )
	{
		// This is the first time we've processed this shape:
		continueData->m_shapeKeys.expandOne() = shapeContainer->getFirstKey();
	}
	hkpShapeKey key = continueData->m_shapeKeys[branchDepth];
	hkpShapeBuffer buffer;

	while ( ( key != HK_INVALID_SHAPE_KEY ) && ( numSimpleShapes > 0 ) )
	{
		const hkpShape* child = shapeContainer->getChildShape(key, buffer );
		if ( buildPartialShapeDisplay( child, transform, branchDepth + 1, numSimpleShapes, continueData, displayGeometries, scale ) )
		{
			// We successfully processed the subshape, so try the next one.
			key = shapeContainer->getNextKey( key );
			continueData->m_shapeKeys[branchDepth] = key;
		}
		else
		{
			return false;
		}
	}

	if ( key == HK_INVALID_SHAPE_KEY )
	{
		// We finished processing this subshape.
		continueData->m_shapeKeys.popBack();
		return true;
	}
	else
	{
		return false;
	}
}

hkBool hkpShapeDisplayBuilder::buildPartialShapeDisplay_SampledHeightField( const hkpSampledHeightFieldShape* heightField, const hkTransform& transform, int branchDepth, int& numSimpleShapes, hkpShapeContinueData* continueData, hkArray<hkDisplayGeometry*>& displayGeometries )
{
	HK_ASSERT2( 0x3e421e73, numSimpleShapes > 0, "Cannot build a shape display for 0 numSimpleShapes." );
	HK_ASSERT2( 0x3e421e82, branchDepth == continueData->m_shapeKeys.getSize(), "The continue data is inconsistent with the traversal of the shape hierarchy." );

	hkDisplayGeometry* displayGeom = getCurrentRawGeometry(displayGeometries);
	hkGeometry* geom = displayGeom->getGeometry();
	HK_ASSERT(0x34673afe, geom != HK_NULL);

	// Convert these vertices to the transformed space.
	hkVector4 scale = heightField->m_intToFloatScale;

	if ( continueData->m_i == -1 )
	{
		// This is the first time we've processed this shape:
		continueData->m_i = 0;
	}

	while ( ( continueData->m_i < heightField->m_xRes - 1 ) && ( numSimpleShapes > 0 ) )
	{
		if ( continueData->m_j == -1 )
		{
			// This is the first time we've processed this row:
			continueData->m_j = 0;
		}

		while ( ( continueData->m_j < heightField->m_zRes - 1 ) && ( numSimpleShapes > 0 ) )
		{
			const int& i = continueData->m_i;
			const int& j = continueData->m_j;

			hkVector4 p00; p00.set( i+0.f, heightField->getHeightAt( i+0, j+0 ), j+0.f ); p00.mul( scale );
			hkVector4 p01; p01.set( i+0.f, heightField->getHeightAt( i+0, j+1 ), j+1.f ); p01.mul( scale );
			hkVector4 p10; p10.set( i+1.f, heightField->getHeightAt( i+1, j+0 ), j+0.f ); p10.mul( scale );
			hkVector4 p11; p11.set( i+1.f, heightField->getHeightAt( i+1, j+1 ), j+1.f ); p11.mul( scale );

			{
				int vertBase = geom->m_vertices.getSize();

				hkVector4 *const vertices = geom->m_vertices.expandBy( 4 );
				vertices[0].setTransformedPos(transform, p00 );
				vertices[1].setTransformedPos(transform, p01 );
				vertices[2].setTransformedPos(transform, p10 );
				vertices[3].setTransformedPos(transform, p11 );

				if ( heightField->getTriangleFlip())
				{
					hkGeometry::Triangle *const triangles = geom->m_triangles.expandBy( 2 );
					triangles[0].set(vertBase + 0, vertBase + 1, vertBase + 3);
					triangles[1].set(vertBase + 0, vertBase + 3, vertBase + 2);
				}
				else
				{
					hkGeometry::Triangle *const triangles = geom->m_triangles.expandBy( 2 );
					triangles[0].set(vertBase + 0, vertBase + 1, vertBase + 2);
					triangles[1].set(vertBase + 3, vertBase + 2, vertBase + 1);
				}

				++continueData->m_j;
				--numSimpleShapes;
			}
		}

		if ( continueData->m_j == heightField->m_zRes - 1 )
		{
			// We've finished processing the row.
			continueData->m_j = -1;
		}
		else
		{
			return false;
		}

		++continueData->m_i;
	}

	if ( continueData->m_i == heightField->m_xRes - 1 )
	{
		// We've finished processing this shape.
		continueData->m_i = -1;
		return true;
	}
	else
	{
		return false;
	}
}

hkBool hkpShapeDisplayBuilder::buildPartialShapeDisplay( const hkpShape* shape, const hkTransform& transform, 
														 int branchDepth, int& numSimpleShapes, 
														 hkpShapeContinueData* continueData, 
														 hkArray<hkDisplayGeometry*>& displayGeometries,
														 const hkVector4* scale )
{
	HK_ASSERT2( 0x3e421e73, numSimpleShapes > 0, "Cannot build a shape display for 0 numSimpleShapes." );

	switch( shape->getType() )
	{
		case hkcdShapeType::SPHERE:
		{
			const hkpSphereShape* sphereShape = static_cast<const hkpSphereShape*>(shape);
			buildShapeDisplay_Sphere( sphereShape, transform, displayGeometries, scale );
			--numSimpleShapes;
			break;
		}
		case hkcdShapeType::MULTI_SPHERE:
		{
			HK_WARN_ON_DEBUG_IF(scale, 0x5ba9c97f, "Scale is not supported for this shape type and will be ignored");
			const hkpMultiSphereShape* s = static_cast<const hkpMultiSphereShape*>(shape);
			buildPartialShapeDisplay_MultiSphere( s, transform, branchDepth, numSimpleShapes, continueData, displayGeometries );
			break;
		}
		case hkcdShapeType::PLANE:
		{
			HK_WARN_ON_DEBUG_IF(scale, 0x5ba9c97f, "Scale is not supported for this shape type and will be ignored");
			const hkpPlaneShape* planeShape = static_cast<const hkpPlaneShape*>(shape);
			buildShapeDisplay_Plane( planeShape, transform, displayGeometries );
			--numSimpleShapes;
			break;
		}
		case hkcdShapeType::CAPSULE:
		{
			const hkpCapsuleShape* s = static_cast<const hkpCapsuleShape*>(shape);
			buildShapeDisplay_Capsule( s, transform, displayGeometries, scale );
			--numSimpleShapes;
			break;
		}
		case hkcdShapeType::CYLINDER:
		{
			const hkpCylinderShape* s = static_cast<const hkpCylinderShape*>(shape);
			buildShapeDisplay_Cylinder( s, transform, displayGeometries, scale );
			--numSimpleShapes;
			break;
		}
		case hkcdShapeType::MULTI_RAY:
		{
			HK_WARN_ON_DEBUG_IF(scale, 0x5ba9c97f, "Scale is not supported for this shape type and will be ignored");
			const hkpMultiRayShape* s = static_cast<const hkpMultiRayShape*>(shape);
			return buildPartialShapeDisplay_MultiRay( s, transform, branchDepth, numSimpleShapes, continueData, displayGeometries );
		}
		case hkcdShapeType::BOX:
		{
			const hkpBoxShape* boxShape = static_cast<const hkpBoxShape*>(shape);
			buildShapeDisplay_Box( boxShape, transform, displayGeometries, scale );
			--numSimpleShapes;
			break;
		}
		case hkcdShapeType::CONVEX_TRANSLATE:
		{
			const hkpConvexTranslateShape* ts = static_cast<const hkpConvexTranslateShape*>( shape );

			// Obtain translation
			hkVector4 translation = ts->getTranslation();
			if (scale)
			{
				translation.mul(*scale);				
			}

			// Compute child transform
			hkTransform childTransform; 
			{			
				hkTransform translateTransform; translateTransform.setIdentity();	
				translateTransform.setTranslation(translation);
				childTransform.setMul(transform, translateTransform);
			}  

			return buildPartialShapeDisplay( ts->getChildShape(), childTransform, branchDepth, numSimpleShapes, continueData, displayGeometries, scale );
		}
		case hkcdShapeType::CONVEX_TRANSFORM:
		{
			HK_WARN_ON_DEBUG_IF(scale, 0x5ba9c97f, "Scale is not supported for this shape type and will be ignored");
			const hkpConvexTransformShape* convexTransformShape = static_cast<const hkpConvexTransformShape*>( shape );
			const hkQsTransform& convexTransform = convexTransformShape->getQsTransform();
			hkTransform convexTransformNoScale; convexTransform.copyToTransformNoScale(convexTransformNoScale);				
			hkTransform childTransform; childTransform.setMul(transform, convexTransformNoScale);
			return buildPartialShapeDisplay( convexTransformShape->getChildShape(), childTransform, branchDepth, 
											 numSimpleShapes, continueData, displayGeometries, &convexTransform.m_scale );
		}	
		case hkcdShapeType::TRANSFORM:
		{
			HK_WARN_ON_DEBUG_IF(scale, 0x5ba9c97f, "Scale is not supported for this shape type and will be ignored");
			const hkpTransformShape* ts = static_cast<const hkpTransformShape*>( shape );
			hkTransform T; T.setMul( transform, ts->getTransform() );
			return buildPartialShapeDisplay( ts->getChildShape(), T, branchDepth, numSimpleShapes, continueData, displayGeometries );
		}
		case hkcdShapeType::BV:
		{
			HK_WARN_ON_DEBUG_IF(scale, 0x5ba9c97f, "Scale is not supported for this shape type and will be ignored");
			const hkpBvShape* bvShape = static_cast<const hkpBvShape*>(shape);
			return buildPartialShapeDisplay( bvShape->getBoundingVolumeShape(), transform, branchDepth, numSimpleShapes, continueData, displayGeometries );
		}
		case hkcdShapeType::MOPP:
		{
			HK_WARN_ON_DEBUG_IF(scale, 0x5ba9c97f, "Scale is not supported for this shape type and will be ignored");
			const hkpMoppBvTreeShape* bvShape = static_cast<const hkpMoppBvTreeShape*>(shape);
			return buildPartialShapeDisplay( bvShape->getChild(), transform, branchDepth, numSimpleShapes, continueData, displayGeometries );
		}
		case hkcdShapeType::LIST:
		case hkcdShapeType::BV_TREE:
		case hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_BV_TREE:
		case hkcdShapeType::CONVEX_LIST:
		case hkcdShapeType::COLLECTION:
		case hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_COLLECTION:
		case hkcdShapeType::TRIANGLE_COLLECTION:
		case hkcdShapeType::EXTENDED_MESH:
		case hkcdShapeType::COMPRESSED_MESH:
		case hkcdShapeType::BV_COMPRESSED_MESH:
		case hkcdShapeType::STATIC_COMPOUND:
		{
			const hkpShapeContainer* shapeContainer = shape->getContainer();
			return buildPartialShapeDisplay_ShapeContainer( shapeContainer, transform, branchDepth, numSimpleShapes, 
															continueData, displayGeometries, scale );
		}
		case hkcdShapeType::TRIANGLE:
		{
			const hkpTriangleShape* triangleShape = static_cast<const hkpTriangleShape*>(shape);
			buildShapeDisplay_Triangle( triangleShape, transform, displayGeometries, scale );
			--numSimpleShapes;
			break;
		}
		case hkcdShapeType::CONVEX_VERTICES:
		{
			const hkpConvexVerticesShape* cvShape = static_cast<const hkpConvexVerticesShape*>(shape);			
			buildShapeDisplay_ConvexVertices( cvShape, transform, displayGeometries, scale );
			--numSimpleShapes;
			break;
		}

		case hkcdShapeType::CONVEX_PIECE:
		{
			HK_WARN_ON_DEBUG_IF(scale, 0x5ba9c97f, "Scale is not supported for this shape type and will be ignored");
			const hkpConvexPieceShape* triangulatedConvexShape = static_cast<const hkpConvexPieceShape*>(shape);
			// This shape is deprecated, so we haven't implemented partial building for it.
			buildShapeDisplay_ConvexPiece( triangulatedConvexShape, transform, displayGeometries );
			--numSimpleShapes;
			break;
		}
		case hkcdShapeType::SAMPLED_HEIGHT_FIELD:
		{
			HK_WARN_ON_DEBUG_IF(scale, 0x5ba9c97f, "Scale is not supported for this shape type and will be ignored");
			const hkpSampledHeightFieldShape* heightField = static_cast<const hkpSampledHeightFieldShape*>(shape);
			return buildPartialShapeDisplay_SampledHeightField( heightField, transform, branchDepth, numSimpleShapes, continueData, displayGeometries );
		}
		default:
		{
			HK_WARN_ON_DEBUG_IF(scale, 0x5ba9c97f, "Scale is not supported for this shape type and will be ignored");
			if ( buildShapeDisplay_UserShapes( shape, transform, displayGeometries ) )
			{
				--numSimpleShapes;
			}
			break;
		}
	}
	return true;
}

hkReferencedObject* hkpShapeDisplayBuilder::getInitialContinueData( const hkReferencedObject* source )
{
	return new hkpShapeContinueData();
}


hkDisplayGeometry* hkpShapeDisplayBuilder::getCurrentRawGeometry(hkArray<hkDisplayGeometry*>& displayGeometries)
{
	if (m_currentGeometry == HK_NULL)
	{
		hkGeometry* geom = new hkGeometry;
		m_currentGeometry = new hkDisplayConvex(geom);
		displayGeometries.pushBack(m_currentGeometry);
	}
	return m_currentGeometry;
}


void hkpShapeDisplayBuilder::resetCurrentRawGeometry()
{
	m_currentGeometry = HK_NULL;
}


HK_SINGLETON_IMPLEMENTATION(hkpUserShapeDisplayBuilder);


void hkpUserShapeDisplayBuilder::registerUserShapeDisplayBuilder( ShapeBuilderFunction f, hkpShapeType type )
{
	for (int i = 0; i < m_userShapeBuilders.getSize(); ++i )
	{
		if ( m_userShapeBuilders[i].type == type )
		{
			HK_WARN(0x7bbfa3c4, "You have registered two shape display builders for user type" << type << ". Do you have two different shapes with this type?");
			return;
		}
	}
	UserShapeBuilder b;
	b.f = f;
	b.type = type;

	m_userShapeBuilders.pushBack(b);
}

void HK_CALL hkpShapeDisplayBuilder::addObjectToDebugDisplay( const hkpShape* shape, hkTransform& t, hkUlong id )
{
	hkpShapeDisplayBuilder::hkpShapeDisplayBuilderEnvironment env;
	hkpShapeDisplayBuilder builder(env);


	hkArray<hkDisplayGeometry*> displayGeometries;

	builder.buildDisplayGeometries( shape, displayGeometries );
	hkDebugDisplay::getInstance().addGeometry( displayGeometries, t, id, 0, (hkUlong)shape );

	while( displayGeometries.getSize() )
	{
		delete displayGeometries[0];
		displayGeometries.removeAt(0);
	}
}


void hkpShapeDisplayBuilder::buildDisplayGeometries( const hkReferencedObject* source, hkArray<hkDisplayGeometry*>& displayGeometries )
{
	buildDisplayGeometries( static_cast<const hkpShape*>( source ), displayGeometries );
}

hkBool hkpShapeDisplayBuilder::buildPartialDisplayGeometries( const hkReferencedObject* source, int& numSimpleShapes, hkReferencedObject* continueData, hkArray<hkDisplayGeometry*>& displayGeometries )
{
	HK_ASSERT2( 0x3e421e73, numSimpleShapes > 0, "Cannot build a shape display for 0 numSimpleShapes." );

	hkTransform transform;
	transform.setIdentity();

	resetCurrentRawGeometry();
	displayGeometries.clear();

	const hkpShape *const shape = static_cast<const hkpShape*>( source );
	hkpShapeContinueData *const shapeContinueData = static_cast<hkpShapeContinueData*>( continueData );

	if ( buildPartialShapeDisplay( shape, transform, 0, numSimpleShapes, shapeContinueData, displayGeometries ) )
	{
		// We've finished with this continue data.
		shapeContinueData->removeReference();
		return true;
	}
	else
	{
		return false;
	}
}

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
