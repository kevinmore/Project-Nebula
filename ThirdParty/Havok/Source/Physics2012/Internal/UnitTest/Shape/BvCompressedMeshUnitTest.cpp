/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Internal/hkpInternal.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/Reflection/Registry/hkVtableClassRegistry.h>
#include <Common/Serialize/Util/hkSerializeUtil.h>
#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>
#include <Common/Base/Container/Set/hkSet.h>
#include <Common/Base/Types/Geometry/hkGeometry.h>
#include <Common/Base/Types/Geometry/hkStridedVertices.h>
#include <Common/Base/Types/Geometry/Geometry/hkGeometryUtil.h>
#include <Physics2012/Collide/Filter/Group/hkpGroupFilter.h>
#include <Physics2012/Collide/Query/Collector/RayCollector/hkpAllRayHitCollector.h>
#include <Physics2012/Collide/Shape/Convex/Box/hkpBoxShape.h>
#include <Physics2012/Collide/Shape/Convex/Capsule/hkpCapsuleShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesShape.h>
#include <Physics2012/Collide/Shape/Convex/Cylinder/hkpCylinderShape.h>
#include <Physics2012/Collide/Shape/Convex/Sphere/hkpSphereShape.h>
#include <Physics2012/Collide/Shape/Convex/Triangle/hkpTriangleShape.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastInput.h>
#include <Physics2012/Internal/Collide/BvCompressedMesh/hkpBvCompressedMeshShapeCinfo.h>
#include <Physics2012/Collide/Query/Collector/RayCollector/hkpClosestRayHitCollector.h>

#define COMPARE_EQUAL_AABB(A,B,EPSILON)	((A).m_min.allEqual<3>((B).m_min, hkSimdReal::fromFloat(EPSILON)) && (A).m_max.allEqual<3>((B).m_max, hkSimdReal::fromFloat(EPSILON)))

namespace
{
	class BvCompressedMeshUnitTests
	{
		public:

			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_SHAPE, BvCompressedMeshUnitTests);

			BvCompressedMeshUnitTests();
			void runTests();

		protected:

			// Custom mesh construction info structure used to provide per triangle filter info and user data
			struct MyMeshConstructionInfo : public hkpDefaultBvCompressedMeshShapeCinfo
			{
				HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_SHAPE, BvCompressedMeshUnitTests::MyMeshConstructionInfo);

				MyMeshConstructionInfo(const hkGeometry* geometry = HK_NULL, hkUint32 dataMin = 0, hkUint32 dataMax = 255) 
					: hkpDefaultBvCompressedMeshShapeCinfo(geometry), m_dataMin(dataMin), m_dataRange(dataMax - dataMin + 1) {}

				virtual hkUint32 getTriangleCollisionFilterInfo(int triangleIndex) const 
				{ 
					return m_dataMin + triangleIndex % m_dataRange;
				}

				virtual hkUint32 getTriangleUserData(int triangleIndex) const 
				{ 					
					return m_dataMin + m_dataRange - (triangleIndex % m_dataRange) - 1; 
				}

				virtual hkUint32 getConvexShapeCollisionFilterInfo(int convexIndex) const 
				{ 
					return m_dataMin + convexIndex % m_dataRange;					
				}

				virtual hkUint32 getConvexShapeUserData(int convexIndex) const 
				{ 
					return m_dataMin + m_dataRange - (convexIndex % m_dataRange) - 1; 
				}

				hkUint32 m_dataMin;
				hkUint32 m_dataRange;
			};

		protected:

			void testHkpShape();
			void testHkpShapeCastRay();
			void testHkpShapeCastRayCollectorBoxSimple();
			void testHkpShapeCastRayWithCollector();
			void testHkpShapeGetAabb();
			void testHkpShapeContainer();
			void testHkpBvTreeShape();
			void testConvexShapes();
			void testPerPrimitiveData();
			void testDegeneracies();
			static void createBox(hkVector4Parameter halfExtents, hkGeometry& geometryOut);

		protected:

			hkGeometry m_geometry;
			MyMeshConstructionInfo m_cInfo;
			hkArray<hkUint32> m_triangleToKeyMap;
			hkPointerMap<hkUint32, int> m_keyToTriangleMap;
			hkArray<hkUint32> m_convexShapeToKeyMap;
			hkRefPtr<hkpBvCompressedMeshShape> m_meshShape;
	};
}


BvCompressedMeshUnitTests::BvCompressedMeshUnitTests()
{
	// Create simple box geometry	
	hkVector4 halfExtents; halfExtents.set(1, 2, 3);
	createBox(halfExtents, m_geometry);			

	// Create bv compressed mesh
	new (&m_cInfo) MyMeshConstructionInfo(&m_geometry);
	m_cInfo.m_collisionFilterInfoMode = hkpBvCompressedMeshShape::PER_PRIMITIVE_DATA_PALETTE;
	m_cInfo.m_userDataMode = hkpBvCompressedMeshShape::PER_PRIMITIVE_DATA_PALETTE;
	m_cInfo.m_weldingType = hkpWeldingUtility::WELDING_TYPE_ANTICLOCKWISE;
	m_cInfo.m_triangleIndexToShapeKeyMap = &m_triangleToKeyMap;
	m_meshShape = hkRefNew<hkpBvCompressedMeshShape>(new hkpBvCompressedMeshShape(m_cInfo));

	// Create key to triangle map
	HK_TEST(m_triangleToKeyMap.getSize() == m_geometry.m_triangles.getSize());
	for (int triangleIdx = 0; triangleIdx < m_triangleToKeyMap.getSize(); ++triangleIdx)
	{
		const hkBool isUnique = m_keyToTriangleMap.insert(m_triangleToKeyMap[triangleIdx], triangleIdx);
		HK_TEST(isUnique);
	}
}


void BvCompressedMeshUnitTests::runTests()
{		
	testHkpShape();	
	testHkpShapeContainer();	
	testHkpBvTreeShape();
	testConvexShapes();
	testHkpShapeCastRayCollectorBoxSimple();
	testPerPrimitiveData();
	testDegeneracies();
}

void BvCompressedMeshUnitTests::testDegeneracies()
{
	// Input degenerate triangle
	{
		// We expect this warning, so disable it.
		hkDisableError noPrimitivesLeftAfterWelding(0xC07742D4);

		// Set up a geometry with an outrageously degenerate triangle
		hkGeometry geometry;
		geometry.m_vertices.expandOne().set(0, 0, 0);
		geometry.m_vertices.expandOne().set(1, 0, 0);
		geometry.m_vertices.expandOne().set(0.5f, 0, 0);
		geometry.m_triangles.expandOne().set(0, 1, 2, 0);

		// Create the mesh
		hkpDefaultBvCompressedMeshShapeCinfo meshInfo(&geometry);
		hkArray<hkUint32> triangleToKey;
		meshInfo.m_triangleIndexToShapeKeyMap = &triangleToKey;
		hkpBvCompressedMeshShape* mesh = new hkpBvCompressedMeshShape(meshInfo);

		// Check results
		HK_TEST(triangleToKey[0] == HK_INVALID_SHAPE_KEY);

		mesh->removeReference();
	}

	// Output degenerate triangle
	{		
		// Set up a geometry with a large quantization range so that small differences in vertex coordinates are quantized 
		// into the same value.
		hkGeometry geometry;
		geometry.m_vertices.expandOne().set(0, 0, 0);
		geometry.m_vertices.expandOne().set(0, 100, 0);
		geometry.m_vertices.expandOne().set(0, 0, 100);
		geometry.m_vertices.expandOne().set(0, 0.0005f, 100);
		geometry.m_triangles.expandOne().set(0, 1, 2, 0);
		geometry.m_triangles.expandOne().set(1, 3, 2, 0);

		// Create the mesh
		hkpDefaultBvCompressedMeshShapeCinfo meshInfo(&geometry);		
		hkArray<hkUint32> triangleToKey;
		meshInfo.m_triangleIndexToShapeKeyMap = &triangleToKey;
		hkpBvCompressedMeshShape* mesh = new hkpBvCompressedMeshShape(meshInfo);

		// Check results
		HK_TEST(triangleToKey[1] == HK_INVALID_SHAPE_KEY);

		mesh->removeReference();
	}
}


void BvCompressedMeshUnitTests::testPerPrimitiveData()
{
	// Create a 16X16 grid
	hkGeometry geometry;
	{
		hkGeometryUtil::GridInput input(16, hkVector4::getConstant<HK_QUADREAL_0100>());
		hkGeometryUtil::createGrid(input, &geometry);
	}

	hkRefPtr<hkpConvexShape> convexShape = hkRefNew<hkpConvexShape>(new hkpSphereShape(0.1f));
	hkArray<hkUint32> triangleToKeyMap;
	hkArray<hkUint32> convexToKeyMap;
	hkpBvCompressedMeshShape::PerPrimitiveDataMode dataModes[] = { hkpBvCompressedMeshShape::PER_PRIMITIVE_DATA_NONE, 
																   hkpBvCompressedMeshShape::PER_PRIMITIVE_DATA_8_BIT,
																   hkpBvCompressedMeshShape::PER_PRIMITIVE_DATA_PALETTE };

	// Test all three data modes
	for (int i = 0; i < 3; ++i)
	{			
		hkpBvCompressedMeshShape::PerPrimitiveDataMode const dataMode = dataModes[i];
		hkUint32 hasData = (dataMode == hkpBvCompressedMeshShape::PER_PRIMITIVE_DATA_NONE ? 0 : 0xFFFFFFFF);
		triangleToKeyMap.clear();
		convexToKeyMap.clear();

		// Create BVCMS
		hkRefPtr<hkpBvCompressedMeshShape> bvcms;
		MyMeshConstructionInfo cInfo(&geometry);
		{			
			cInfo.m_collisionFilterInfoMode = dataMode;
			cInfo.m_userDataMode = dataMode;
			cInfo.m_triangleIndexToShapeKeyMap = &triangleToKeyMap;
			cInfo.m_convexShapeIndexToShapeKeyMap = &convexToKeyMap;

			// Add instances of the convex shape in a grid
			hkQsTransform transform; transform.setIdentity();
			for (hkReal z = 0; z < 16; ++z)
			{
				for (hkReal x = 0; x < 16; ++x)
				{
					hkVector4 translation; translation.set(x, 0, z);
					transform.setTranslation(translation);
					cInfo.addConvexShape(convexShape, transform);
				}
			}
			
			bvcms = hkRefNew<hkpBvCompressedMeshShape>(new hkpBvCompressedMeshShape(cInfo));
		}

		// Check collision filter info and user data per triangle
		for (int triangleIndex = 0; triangleIndex < triangleToKeyMap.getSize(); ++triangleIndex)
		{
			const hkUint32 expectedCollisionFilterInfo = cInfo.getTriangleCollisionFilterInfo(triangleIndex) & hasData;
			HK_TEST(bvcms->getCollisionFilterInfo(triangleToKeyMap[triangleIndex]) == expectedCollisionFilterInfo);
			const hkUint32 expectedUserData = cInfo.getTriangleUserData(triangleIndex) & hasData;
			HK_TEST(bvcms->getPrimitiveUserData(triangleToKeyMap[triangleIndex]) == expectedUserData);
		}

		// Check collision filter info and user data per convex shape
		for (int convexIndex = 0; convexIndex < convexToKeyMap.getSize(); ++convexIndex)
		{
			const hkUint32 expectedCollisionFilterInfo = cInfo.getConvexShapeCollisionFilterInfo(convexIndex) & hasData;
			HK_TEST(bvcms->getCollisionFilterInfo(convexToKeyMap[convexIndex]) == expectedCollisionFilterInfo);
			const hkUint32 expectedUserData = cInfo.getConvexShapeUserData(convexIndex) & hasData;
			HK_TEST(bvcms->getPrimitiveUserData(convexToKeyMap[convexIndex]) == expectedUserData);
		}
	}

	// Check asserts
	{
		// Collision filter info out of range
		{		
			convexToKeyMap.clear();
			hkRefPtr<hkpBvCompressedMeshShape> bvcms;
			MyMeshConstructionInfo cInfo(HK_NULL, 256, 256);
			cInfo.m_collisionFilterInfoMode = hkpBvCompressedMeshShape::PER_PRIMITIVE_DATA_8_BIT;
			cInfo.m_userDataMode = hkpBvCompressedMeshShape::PER_PRIMITIVE_DATA_NONE;			
			cInfo.m_convexShapeIndexToShapeKeyMap = &convexToKeyMap;			
			hkQsTransform transform; transform.setIdentity();
			cInfo.addConvexShape(convexShape, transform);			
			HK_TEST_ASSERT_AND_CONTINUE(0x37a7a2bb, bvcms = hkRefNew<hkpBvCompressedMeshShape>(new hkpBvCompressedMeshShape(cInfo)));		
		}		

		// User data out of range
		{		
			convexToKeyMap.clear();
			hkRefPtr<hkpBvCompressedMeshShape> bvcms;
			MyMeshConstructionInfo cInfo(HK_NULL, 256, 256);
			cInfo.m_collisionFilterInfoMode = hkpBvCompressedMeshShape::PER_PRIMITIVE_DATA_NONE;
			cInfo.m_userDataMode = hkpBvCompressedMeshShape::PER_PRIMITIVE_DATA_8_BIT;			
			cInfo.m_convexShapeIndexToShapeKeyMap = &convexToKeyMap;
			hkQsTransform transform; transform.setIdentity();
			cInfo.addConvexShape(convexShape, transform);			
			HK_TEST_ASSERT_AND_CONTINUE(0x37a7a2bb, bvcms = hkRefNew<hkpBvCompressedMeshShape>(new hkpBvCompressedMeshShape(cInfo)));		
		}
	}
}


void BvCompressedMeshUnitTests::testConvexShapes()
{		
	// Create a convex shape of each type	
	hkArray< hkRefPtr<hkpConvexShape> > convexShapes;
	{	
		const hkReal convexRadius = 0.05f;	
		const hkReal oneMinusR = 1 - convexRadius;

		// Sphere
		convexShapes.pushBack(hkRefNew<hkpConvexShape>(new hkpSphereShape(1)));

		// Capsule
		{ 
			hkVector4 vertexA; vertexA.set(0, 1, 0);
			hkVector4 vertexB; vertexB.set(0, -1, 0);
			convexShapes.pushBack(hkRefNew<hkpConvexShape>(new hkpCapsuleShape(vertexA, vertexB, 1)));
		}
		
		// Cylinder
		{
			hkVector4 vertexA; vertexA.set(0, oneMinusR, 0);
			hkVector4 vertexB; vertexB.set(0, -oneMinusR, 0);
			convexShapes.pushBack(hkRefNew<hkpConvexShape>(new hkpCylinderShape(vertexA, vertexB, oneMinusR, convexRadius)));
		}	

		// Box		
		{
			hkVector4 halfExtents; halfExtents.set(oneMinusR, oneMinusR, oneMinusR);
			convexShapes.pushBack(hkRefNew<hkpConvexShape>(new hkpBoxShape(halfExtents, convexRadius)));
		}		

		// Convex vertices
		hkRefPtr<hkpConvexVerticesShape> convexVertices;
		{					
			hkVector4 vertices[8];				
			for (int i = 0; i < 8; ++i)
			{
				vertices[i].set(1.0f - ((i & 1) << 1), 1.0f - (i & 2), 1.0f - ((i & 4) >> 1));					
			}			

			hkStridedVertices stridedVertices(vertices, 8);
			hkpConvexVerticesShape::BuildConfig config;			
			config.m_convexRadius = convexRadius;
			config.m_maxRelativeShrink = 0.5f;
			config.m_maxShrinkingVerticesDisplacement = 1;
			convexShapes.pushBack(hkRefNew<hkpConvexShape>(new hkpConvexVerticesShape(stridedVertices, config)));
		}		
	}	
	
	// Create a BVCMS with the convex shapes with different transforms
	hkpBvCompressedMeshShape* bvcms;
	MyMeshConstructionInfo cInfo(HK_NULL);
	const int numConvexChildren = convexShapes.getSize() * 20;
	const hkReal maxConvexShapeError = 0.01f;
	{			
		cInfo.m_collisionFilterInfoMode = hkpBvCompressedMeshShape::PER_PRIMITIVE_DATA_PALETTE;
		cInfo.m_userDataMode = hkpBvCompressedMeshShape::PER_PRIMITIVE_DATA_PALETTE;
		cInfo.m_weldingType = hkpWeldingUtility::WELDING_TYPE_ANTICLOCKWISE;
		cInfo.m_triangleIndexToShapeKeyMap = &m_triangleToKeyMap;
		cInfo.m_convexShapeIndexToShapeKeyMap = &m_convexShapeToKeyMap;
		cInfo.m_maxConvexShapeError = maxConvexShapeError;

		// Add convex shapes with random transforms		
		hkPseudoRandomGenerator rnd(2);
		hkVector4 minScale; minScale.set(0.05f, 0.05f, 0.05f);
		hkVector4 maxScale; maxScale.set(10, 10, 10);
		hkSimdReal maxTranslation; maxTranslation.setFromFloat(100.0f);
		for (int i = 0; i < numConvexChildren; ++i)
		{
			const hkpConvexShape* convexShape = convexShapes[i % convexShapes.getSize()];
			hkcdShape::ShapeType shapeType = convexShape->getType();

			// Calculate random transform
			hkQsTransform transform;
			{			
				// Scale (only box and convex vertices support non-uniform)
				hkVector4 scale; rnd.getRandomVectorRange(minScale, maxScale, scale);
				hkVector4 sign; rnd.getRandomVector11(sign);
				scale.setFlipSign(scale, sign);
				if (shapeType != hkcdShapeType::BOX && shapeType != hkcdShapeType::CONVEX_VERTICES)
				{
					scale.setAll(scale.getComponent<0>());
				}
				transform.setScale(scale);

				// Translation
				hkVector4 translation;
				rnd.getRandomVector11(translation);
				translation.mul(maxTranslation);
				transform.setTranslation(translation);

				// Rotation
				hkQuaternion rotation;
				rnd.getRandomRotation(rotation);				
				transform.setRotation(rotation);
			}
						
			cInfo.addConvexShape(convexShape, transform);
		}
		
		bvcms = new hkpBvCompressedMeshShape(cInfo);
		HK_TEST(m_convexShapeToKeyMap.getSize() == numConvexChildren);
	}

#if !defined(HK_REAL_IS_DOUBLE)
	// Save and load back to make sure serialization works fine (binary serialization is broken for double builds anyways)
	
	
	hkRefPtr<hkResource> resource;
	{
		// Obtain BVCMS class information
		const hkClass* klass = hkBuiltinTypeRegistry::getInstance().getVtableClassRegistry()->getClassFromVirtualInstance(bvcms);
		HK_ASSERT(0x751bc56b, klass);

		{
			// 1: Serialize to buffer (binary tagfile)
			hkArray<char> buffer;
			hkSerializeUtil::save(bvcms, *klass, hkOstream(buffer).getStreamWriter(), hkSerializeUtil::SAVE_DEFAULT);
			// Load from buffer
			resource = hkRefNew<hkResource>(hkSerializeUtil::load(buffer.begin(), buffer.getSize()));
			bvcms->removeReference();
			bvcms = resource->getContents<hkpBvCompressedMeshShape>();
			HK_TEST2(bvcms != HK_NULL, "Failed loading serialized shape from binary tagfile");
		}
		{
			// 2: Serialize to buffer (xml tagfile)
			hkArray<char> buffer;
			hkSerializeUtil::save(bvcms, *klass, hkOstream(buffer).getStreamWriter(), hkSerializeUtil::SAVE_TEXT_FORMAT);
			// Load from buffer
			resource = hkRefNew<hkResource>(hkSerializeUtil::load(buffer.begin(), buffer.getSize()));
			bvcms = resource->getContents<hkpBvCompressedMeshShape>();
			HK_TEST2(bvcms != HK_NULL, "Failed loading serialized shape from xml tagfile");
		}
		{
			// 3: Serialize to buffer (packfile)
			hkArray<char> buffer;
			hkPackfileWriter::Options options;
			hkSerializeUtil::savePackfile(bvcms, *klass, hkOstream(buffer).getStreamWriter(), options);
			// Load from buffer
			resource = hkRefNew<hkResource>(hkSerializeUtil::load(buffer.begin(), buffer.getSize()));
			bvcms = resource->getContents<hkpBvCompressedMeshShape>();
			HK_TEST2(bvcms != HK_NULL, "Failed loading serialized shape from packfile");
		}
	}
#endif

	// Iterate over all convex children comparing the recovered child shape with the original one
	for (int convexIndex = 0; convexIndex < numConvexChildren; ++convexIndex)
	{			
		// Obtain original convex shape and its transform and create a convex transform shape with them
		const hkpConvexShape* originalShape;
		hkQsTransform convexShapeTransform;
		cInfo.getConvexShape( convexIndex, originalShape, convexShapeTransform );
		const hkcdShape::ShapeType originalType = originalShape->getType();
		hkpConvexTransformShape convexTransform( originalShape, convexShapeTransform );		

		// Obtain child shape from BVCMS
		hkpShapeKey key = m_convexShapeToKeyMap[convexIndex];
		hkpShapeBuffer shapeBuffer;
		const hkpConvexShape* childShape = static_cast<const hkpConvexShape*>(bvcms->getChildShape(key, shapeBuffer));			
		
		// Test data common for all shape types
		{
			HK_TEST( hkMath::equal(childShape->getRadius(), convexTransform.getRadius(), convexTransform.getRadius()*hkReal(0.01f)) );
			HK_TEST(childShape->getUserData() == cInfo.getConvexShapeUserData(convexIndex));				
			HK_TEST(bvcms->getCollisionFilterInfo(key) == cInfo.getConvexShapeCollisionFilterInfo(convexIndex));
		}
		
		// Obtain vertices from the original shape			
		const hkVector4* originalVertices = HK_NULL;
		hkArray<hkVector4> originalVerticesBuffer;
		int numVertices = 0;			
		switch(originalType)
		{
			// Sphere
			case hkcdShapeType::SPHERE:
			{					
				numVertices = 1;
				originalVerticesBuffer.pushBack(hkVector4::getZero());
				originalVertices = originalVerticesBuffer.begin();
				break;
			}

			// Capsule
			case hkcdShapeType::CAPSULE:
			{
				const hkpCapsuleShape* capsule = static_cast<const hkpCapsuleShape*>(originalShape);
				numVertices = 2;
				originalVertices = capsule->getVertices();
				break;					
			}

			// Cylinder
			case hkcdShapeType::CYLINDER:
			{					
				const hkpCylinderShape* cylinder = static_cast<const hkpCylinderShape*>(originalShape);
				numVertices = 2;					
				originalVertices = cylinder->getVertices();

				// Check if cylinder radius has been scaled correctly									
				const hkpCylinderShape* cylinderChild = static_cast<const hkpCylinderShape*>(childShape);
				hkReal scaledRadius = cylinder->getCylinderRadius() * hkMath::abs(convexShapeTransform.getScale()(0));
				HK_TEST( hkMath::equal(cylinderChild->getCylinderRadius(), scaledRadius, scaledRadius*hkReal(0.01f)) );
				break;
			}

			// Box
			case hkcdShapeType::BOX:
			{
				const hkpBoxShape* box = static_cast<const hkpBoxShape*>(originalShape);
				numVertices = 8;					
				originalVerticesBuffer.setSize(8);
				hkpVertexId vertexIds[8] = { 7, 6, 5, 3, 4, 1, 2, 0 };
				box->convertVertexIdsToVertices(vertexIds, 8, (hkcdVertex*)originalVerticesBuffer.begin());
				originalVertices = originalVerticesBuffer.begin();										
				break;
			}

			// Convex vertices
			case hkcdShapeType::CONVEX_VERTICES:
			{
				const hkpConvexVerticesShape* convexVertices = static_cast<const hkpConvexVerticesShape*>(originalShape);					
				convexVertices->getOriginalVertices(originalVerticesBuffer);
				numVertices = originalVerticesBuffer.getSize();
				originalVertices = originalVerticesBuffer.begin();
				break;
			}

			default:
			{					
				HK_TEST2(0, "Unsupported shape type");
			}
		}

		// Recover vertices from the child shape
		hkArray<hkVector4> recoveredVertices;
		if (originalType != hkcdShapeType::CYLINDER)
		{
			const hkpConvexVerticesShape* convexVerticesChild = static_cast<const hkpConvexVerticesShape*>(childShape);
			convexVerticesChild->getOriginalVertices(recoveredVertices);
		}		
		else
		{
			const hkpCylinderShape* cylinderChild = static_cast<const hkpCylinderShape*>(childShape);
			recoveredVertices.pushBack(cylinderChild->getVertex(0));
			recoveredVertices.pushBack(cylinderChild->getVertex(1));
		}
		HK_TEST(recoveredVertices.getSize() == numVertices);					

		// If the original shape was a box only 4 vertices are stored in the bvcms and the rest are computed from those.
		// This may lead to errors above maxConvexShapeError in computed ones.
		hkSimdReal epsilon; 
		epsilon.setFromFloat(originalType == hkcdShapeType::BOX ? 4 * maxConvexShapeError : maxConvexShapeError);

		// Compare transformed original vertices with recovered ones								
		for (int i = 0; i < numVertices; ++i)
		{				
			hkVector4 transformedVertex;
			convexTransform.transformVertex(originalVertices[i], &transformedVertex);
			hkStringBuf text;
			text.printf("Shape %d, vertex %d", convexIndex, i);
			HK_TEST2(transformedVertex.allEqual<3>(recoveredVertices[i], epsilon), text);
		}
	}	
#if defined(HK_REAL_IS_DOUBLE)
	bvcms->removeReference();
#endif
}


void BvCompressedMeshUnitTests::testHkpBvTreeShape()
{
	const hkpBvTreeShape* bvTree = m_meshShape;

	// queryAabb
	{
		hkAabb aabb;
		hkArray<hkpShapeKey> hits;

		// No hits
		{								
			aabb.setEmpty();
			hits.clear();
			bvTree->queryAabb(aabb, hits);
			HK_TEST(hits.getSize() == 0);
			
			aabb.m_min.set(-10, -10, -10);
			aabb.m_max.set(-9, -9, -9);
			hits.clear();
			bvTree->queryAabb(aabb, hits);
			HK_TEST(hits.getSize() == 0);
		}

		// All hits
		{
			aabb.setFull();
			hits.clear();
			bvTree->queryAabb(aabb, hits);
			HK_TEST(hits.getSize() == m_triangleToKeyMap.getSize());
			for (int i = 0; i < hits.getSize(); ++i)
			{
				HK_TEST(m_keyToTriangleMap.hasKey(hits[i]));
			}			

			bvTree->getAabb(hkTransform::getIdentity(), 0, aabb);
			hits.clear();
			bvTree->queryAabb(aabb, hits);
			HK_TEST(hits.getSize() == m_triangleToKeyMap.getSize());
			for (int i = 0; i < hits.getSize(); ++i)
			{
				HK_TEST(m_keyToTriangleMap.hasKey(hits[i]));
			}
		}

		// Some hits
		{
			aabb.m_min.set(-1.1f, -0.1f, -0.1f);
			aabb.m_max.set(-0.9f, -0.1f, -0.1f);
			hits.clear();
			bvTree->queryAabb(aabb, hits);
			HK_TEST(hits.getSize() == 2);
			if (hits[0] == m_triangleToKeyMap[4])
			{
				HK_TEST(hits[1] == m_triangleToKeyMap[5]);
			}
			else
			{
				HK_TEST(hits[0] == m_triangleToKeyMap[5]);
				HK_TEST(hits[1] == m_triangleToKeyMap[4]);
			}
		}
	}

	// queryAabbImpl
	{
		hkAabb aabb;
		hkpShapeKey hits[9];		
		const int maxNumHits = 8;
		hits[maxNumHits] = HK_INVALID_SHAPE_KEY;

		// No hits
		{
			aabb.setEmpty();
			int numHits = bvTree->queryAabbImpl(aabb, hits, maxNumHits);
			HK_TEST(numHits == 0);
		}

		// All hits
		{
			aabb.setFull();
			int numHits = bvTree->queryAabbImpl(aabb, hits, maxNumHits);
			HK_TEST(numHits == m_triangleToKeyMap.getSize());

			// Check that we have maxNumHits different valid hits
			hkSet<int> trianglesFound;
			int triangleIdx = -1;
			for (int i = 0; i < maxNumHits; ++i)
			{				
				HK_TEST(m_keyToTriangleMap.get(hits[i], &triangleIdx) == HK_SUCCESS);				
				HK_TEST(trianglesFound.insert(triangleIdx));
			}

			// Check that no more than maxNumHits values have been written to the shape key array
			HK_TEST(hits[maxNumHits] == HK_INVALID_SHAPE_KEY);
		}

#if defined(HK_PLATFORM_SPU)
		// getContainerImpl
		{	
			hkpShapeBuffer shapeBuffer;
			HK_TEST(bvTree->getContainerImpl(bvTree, shapeBuffer) == bvTree);
		}
#endif
	}
}


void BvCompressedMeshUnitTests::testHkpShape()
{
	const hkpShape* shape = m_meshShape;

	// hkpShape::getContainer
	{
		const hkpShapeContainer* container = shape->getContainer();
		HK_TEST(container == static_cast<hkpShapeContainer*>(m_meshShape));
	}

	// hkpShape::isConvex
	{
		HK_TEST(!shape->isConvex());
	}

	// hkpShape::calcSizeForSpu
	{
		hkpShape::CalcSizeForSpuInput input;

		input.m_isFixedOrKeyframed = true;
		input.m_midphaseAgent3Registered = true;
		input.m_hasDynamicMotionSaved = true;
		HK_TEST(shape->calcSizeForSpu(input, 0) == sizeof(hkpBvCompressedMeshShape));

		input.m_isFixedOrKeyframed = false;
		HK_TEST(shape->calcSizeForSpu(input, 0) == sizeof(hkpBvCompressedMeshShape));
	}

	// hkpShape::getAabb
	testHkpShapeGetAabb();

	// hkpShape::castRay
	testHkpShapeCastRay();

	// hkpShape::castRayWithCollector
	testHkpShapeCastRayWithCollector();
}


void BvCompressedMeshUnitTests::testHkpShapeContainer()
{
	const hkpShapeContainer* shapeContainer = m_meshShape;		

	// getNumChildShapes
	HK_TEST(shapeContainer->getNumChildShapes() == 12);

	// getFirstKey/getNextKey	
	{						
		int numKeys = 0;
		for (hkpShapeKey key = shapeContainer->getFirstKey(); key != HK_INVALID_SHAPE_KEY; key = shapeContainer->getNextKey(key))
		{
			// Check if it is a valid key
			HK_TEST(m_keyToTriangleMap.hasKey(key));
			numKeys++;
		}
		HK_TEST(numKeys == 12);	
	}

	// getChildShape
	{		
		for (int i = 0; i < m_triangleToKeyMap.getSize(); ++i)
		{
			hkpShapeBuffer shapeBuffer;
			const hkpTriangleShape* triangle = static_cast<const hkpTriangleShape*>(shapeContainer->getChildShape(m_triangleToKeyMap[i], shapeBuffer));
			HK_TEST(triangle->getType() == hkcdShapeType::TRIANGLE);

			// Compare recovered vertices against original triangle. When can assume that the vertex order has not 
			// been changed because using welding forces it.
			const hkVector4* vertices = triangle->getVertices();
			hkVector4 originalVertices[3];
			m_geometry.getTriangle(i, originalVertices);
			for (int j = 0; j < 3; ++j)
			{
				HK_TEST(vertices[j].allEqual<3>(originalVertices[j], hkSimdReal::fromFloat(1e-5f)));
			}

			// Check other shape info
			HK_TEST(triangle->getWeldingType() == hkpWeldingUtility::WELDING_TYPE_ANTICLOCKWISE);
			HK_TEST(triangle->getUserData() == m_cInfo.getTriangleUserData(i));
		}
	}

	// getCollisionFilterInfo
	{
		for (int triangleIndex = 0; triangleIndex < m_triangleToKeyMap.getSize(); ++triangleIndex)
		{
			HK_TEST(shapeContainer->getCollisionFilterInfo(m_triangleToKeyMap[triangleIndex]) == m_cInfo.getTriangleCollisionFilterInfo(triangleIndex));			
		}
	}

	// isWeldingEnabled
	HK_TEST(shapeContainer->isWeldingEnabled() == true);
}


void BvCompressedMeshUnitTests::testHkpShapeCastRayWithCollector()
{	
	const hkpShape* shape = m_meshShape;

	// Hit, identity transform
	{		
		hkpShapeRayCastInput input;
		input.m_from.set(-10, 1, 0);
		input.m_to.set(10, 1, 0);		
		const hkTransform& transform = hkTransform::getIdentity();
		hkpCollidable collidable(shape, &transform);		
		hkpAllRayHitCollector collector;
		
		shape->castRayWithCollector(input, collidable, collector);
		collector.sortHits();
		const hkArray<hkpWorldRayCastOutput>& hits = collector.getHits();		
		HK_TEST(hits.getSize() == 2);
		if (hits.getSize() == 2)
		{
			// Check first hit
			{							
				const hkpWorldRayCastOutput& hit = hits[0];
				hkVector4 expectedNormal; expectedNormal.set(-1, 0, 0);

				HK_TEST(hit.hasHit());
				HK_TEST(hkMath::equal(hit.m_hitFraction, 9 / 20.0f));
				HK_TEST(hit.m_normal.allEqual<3>(expectedNormal, hkSimdReal::fromFloat(1e-5f)));
				HK_TEST(hit.m_shapeKeys[0] == m_triangleToKeyMap[5]);
				HK_TEST(hit.m_shapeKeys[1] == HK_INVALID_SHAPE_KEY);
				HK_TEST(hit.m_rootCollidable == &collidable);				
			}

			// Check second hit
			{			
				const hkpWorldRayCastOutput& hit = hits[1];
				hkVector4 expectedNormal; expectedNormal.set(-1, 0, 0);

				HK_TEST(hit.hasHit());
				HK_TEST(hkMath::equal(hit.m_hitFraction, 11 / 20.0f));
				HK_TEST(hit.m_normal.allEqual<3>(expectedNormal, hkSimdReal::fromFloat(1e-5f)));
				HK_TEST(hit.m_shapeKeys[0] == m_triangleToKeyMap[7]);
				HK_TEST(hit.m_shapeKeys[1] == HK_INVALID_SHAPE_KEY);
				HK_TEST(hit.m_rootCollidable == &collidable);
			}
		}							
	}	

	// Hit, non-identity transform
	{		
		hkpShapeRayCastInput input;
		input.m_from.set(-10, 1, 0);
		input.m_to.set(10, 1, 0);
		hkVector4 axis; axis.set(0, 0, 1);
		hkQuaternion rotation; rotation.setAxisAngle(axis, 0.25f * HK_REAL_PI);
		hkTransform transform; transform.set(rotation, hkVector4::getZero());
		hkpCollidable collidable(shape, &transform);
		hkpAllRayHitCollector collector;

		shape->castRayWithCollector(input, collidable, collector);
		collector.sortHits();
		const hkArray<hkpWorldRayCastOutput>& hits = collector.getHits();		
		HK_TEST(hits.getSize() == 2);
		if (hits.getSize() == 2)
		{
			// Check first hit
			{							
				const hkpWorldRayCastOutput& hit = hits[0];
				hkVector4 expectedNormal; expectedNormal.set(-1, 0, 0);
				expectedNormal.setRotatedDir(rotation, expectedNormal);

				HK_TEST(hit.hasHit());
				HK_TEST(hkMath::equal(hit.m_hitFraction, 9 / 20.0f));
				HK_TEST(hit.m_normal.allEqual<3>(expectedNormal, hkSimdReal::fromFloat(1e-5f)));
				HK_TEST(hit.m_shapeKeys[0] == m_triangleToKeyMap[5]);
				HK_TEST(hit.m_shapeKeys[1] == HK_INVALID_SHAPE_KEY);
				HK_TEST(hit.m_rootCollidable == &collidable);				
			}

			// Check second hit
			{			
				const hkpWorldRayCastOutput& hit = hits[1];
				hkVector4 expectedNormal; expectedNormal.set(-1, 0, 0);
				expectedNormal.setRotatedDir(rotation, expectedNormal);

				HK_TEST(hit.hasHit());
				HK_TEST(hkMath::equal(hit.m_hitFraction, 11 / 20.0f));
				HK_TEST(hit.m_normal.allEqual<3>(expectedNormal, hkSimdReal::fromFloat(1e-5f)));
				HK_TEST(hit.m_shapeKeys[0] == m_triangleToKeyMap[7]);
				HK_TEST(hit.m_shapeKeys[1] == HK_INVALID_SHAPE_KEY);
				HK_TEST(hit.m_rootCollidable == &collidable);
			}
		}							
	}
}


void BvCompressedMeshUnitTests::testHkpShapeGetAabb()
{	
	const hkpShape* shape = m_meshShape;

	// No transform, no tolerance
	{
		hkAabb aabb;
		hkTransform transform; transform.setIdentity();
		const hkReal tolerance = 0;
		hkAabb expectedAabb;
		expectedAabb.m_min.set(-1, -2, -3);
		expectedAabb.m_max.set(1, 2, 3);

		shape->getAabb(transform, tolerance, aabb);				
		HK_TEST(COMPARE_EQUAL_AABB(aabb, expectedAabb, 1e-5f));
	}

	// No transform, tolerance
	{
		hkAabb aabb;
		hkTransform transform; transform.setIdentity();
		const hkReal tolerance = 1;
		hkAabb expectedAabb;
		expectedAabb.m_min.set(-2, -3, -4);
		expectedAabb.m_max.set(2, 3, 4);

		shape->getAabb(transform, tolerance, aabb);
		HK_TEST(COMPARE_EQUAL_AABB(aabb, expectedAabb, 1e-5f));				
	}

	// Transform, no tolerance
	{
		hkAabb aabb;
		hkVector4 axis; axis.set(0, 0, 1);
		hkRotation rotation; rotation.setAxisAngle(axis, HK_REAL_DEG_TO_RAD * 90);			
		hkVector4 translation; translation.set(2, 1, 3);				
		hkTransform transform; transform.set(rotation, translation);
		const hkReal tolerance = 0;
		hkAabb expectedAabb;
		expectedAabb.m_min.set(0, 0, 0);
		expectedAabb.m_max.set(4, 2, 6);

		shape->getAabb(transform, tolerance, aabb);
		HK_TEST(COMPARE_EQUAL_AABB(aabb, expectedAabb, 1e-5f));
	}

	// Transform and tolerance
	{
		hkAabb aabb;
		hkVector4 axis; axis.set(0, 0, 1);
		hkRotation rotation; rotation.setAxisAngle(axis, HK_REAL_DEG_TO_RAD * 90);			
		hkVector4 translation; translation.set(2, 1, 3);
		hkTransform transform; transform.set(rotation, translation);
		const hkReal tolerance = 1;
		hkAabb expectedAabb;
		expectedAabb.m_min.set(-1, -1, -1);
		expectedAabb.m_max.set(5, 3, 7);

		shape->getAabb(transform, tolerance, aabb);
		HK_TEST(COMPARE_EQUAL_AABB(aabb, expectedAabb, 1e-5f));
	}						
}

void BvCompressedMeshUnitTests::testHkpShapeCastRayCollectorBoxSimple()
{
	hkpDefaultBvCompressedMeshShapeCinfo info;
	hkVector4 half; half.set(0.5f, 0.5f, 0.5f);
	const hkpBoxShape box(half);
	hkQsTransform transform;
	transform.setIdentity();

	// add box 
	hkVector4 position; position.set( -2.0f, 4.3f, -4.78f );
	transform.setTranslation( position );
	info.addConvexShape( &box, transform );

	// setup the bv mesh
	hkpBvCompressedMeshShape bvmesh(info);

	// shoot ray on the box
	hkpShapeRayCastInput ray;
	ray.m_from.set( 0.0f, 0.0f, 0.0f );
	ray.m_to.set( -2.0f, 4.3f, -4.78f );

	hkpClosestRayHitCollector collector; 	
	hkpCollidable collidable(&bvmesh, &hkTransform::getIdentity());
	bvmesh.castRayWithCollector( ray, collidable, collector );

	HK_TEST( collector.hasHit() );
}

void BvCompressedMeshUnitTests::testHkpShapeCastRay()
{
	const hkpShape* shape = m_meshShape;

	// Miss
	{		
		hkpShapeRayCastInput input;
		input.m_from.set(-10, 5, 0);
		input.m_to.set(10, 5, 0);				

		hkpShapeRayCastOutput output;
		hkBool hasHit = shape->castRay(input, output);
		HK_TEST(!hasHit);							
	}

	// Hit, no filter
	{		
		hkpShapeRayCastInput input;
		input.m_from.set(-10, 1, 0);
		input.m_to.set(10, 1, 0);
		hkVector4 expectedNormal; expectedNormal.set(-1, 0, 0);

		hkpShapeRayCastOutput output;
		hkBool hasHit = shape->castRay(input, output);
		HK_TEST(hasHit);
		HK_TEST(hkMath::equal(output.m_hitFraction, 9 / 20.0f));
		HK_TEST(output.m_normal.allEqual<3>(expectedNormal, hkSimdReal::fromFloat(1e-5f)));
		HK_TEST(output.m_shapeKeys[0] == m_triangleToKeyMap[5]);
		HK_TEST(output.m_shapeKeys[1] == HK_INVALID_SHAPE_KEY);				
	}

	// Hit, group filter
	{		
		hkpShapeRayCastInput input;
		input.m_from.set(-10, 1, 0);
		input.m_to.set(10, 1, 0);				
		input.m_filterInfo = hkpGroupFilter::calcFilterInfo(1);
		hkVector4 expectedNormal; expectedNormal.set(-1, 0, 0);

		// Set up group filter
		hkpGroupFilter filter;
		filter.disableCollisionsBetween(1, 1);
		filter.enableCollisionsBetween(1, 2);
		input.m_rayShapeCollectionFilter = &filter;

		// Set up collision filter info in the mesh shape so the first box face hit is filtered out
		{
			hkArray<hkUint32>& filterInfoPalette = m_meshShape->accessCollisionFilterInfoPalette();
			filterInfoPalette[5] = hkpGroupFilter::calcFilterInfo(1);
			filterInfoPalette[7] = hkpGroupFilter::calcFilterInfo(2);
		}

		hkpShapeRayCastOutput output;
		hkBool hasHit = shape->castRay(input, output);
		HK_TEST(hasHit);
		HK_TEST(hkMath::equal(output.m_hitFraction, 11 / 20.0f));
		HK_TEST(output.m_normal.allEqual<3>(expectedNormal, hkSimdReal::fromFloat(1e-5f)));
		HK_TEST(output.m_shapeKeys[0] == m_triangleToKeyMap[7]);
		HK_TEST(output.m_shapeKeys[1] == HK_INVALID_SHAPE_KEY);

		// Restore original palette values
		{
			hkArray<hkUint32>& filterInfoPalette = m_meshShape->accessCollisionFilterInfoPalette();
			filterInfoPalette[5] = hkpGroupFilter::calcFilterInfo(5);
			filterInfoPalette[7] = hkpGroupFilter::calcFilterInfo(7);
		}
	}
}


void BvCompressedMeshUnitTests::createBox(hkVector4Parameter halfExtents, hkGeometry& geometryOut)
{	
	hkVector4* vertices = geometryOut.m_vertices.expandBy(8);

	// Front face vertices indexes (seen from +Z with +Y as up direction)
	// 0 - 1
	// |   |
	// 3 - 2
	vertices[0].set(-halfExtents(0), halfExtents(1), halfExtents(2));
	vertices[1].set(halfExtents(0), halfExtents(1), halfExtents(2));
	vertices[2].set(halfExtents(0), -halfExtents(1), halfExtents(2));
	vertices[3].set(-halfExtents(0), -halfExtents(1), halfExtents(2));

	// Back face vertices indexes (seen from +Z with +Y as up direction)
	// 4 - 5
	// |   |
	// 7 - 6
	vertices[4].set(-halfExtents(0), halfExtents(1), -halfExtents(2));
	vertices[5].set(halfExtents(0), halfExtents(1), -halfExtents(2));
	vertices[6].set(halfExtents(0), -halfExtents(1), -halfExtents(2));
	vertices[7].set(-halfExtents(0), -halfExtents(1), -halfExtents(2));

	hkArray<hkGeometry::Triangle>& triangles = geometryOut.m_triangles;

	// Front face
	triangles.expandBy(1)->set(3, 2, 1);
	triangles.expandBy(1)->set(3, 1, 0);

	// Back face
	triangles.expandBy(1)->set(6, 7, 4);
	triangles.expandBy(1)->set(6, 4, 5);

	// Left face
	triangles.expandBy(1)->set(4, 7, 3);
	triangles.expandBy(1)->set(4, 3, 0);

	// Right face
	triangles.expandBy(1)->set(2, 6, 5);
	triangles.expandBy(1)->set(2, 5, 1);

	// Bottom face
	triangles.expandBy(1)->set(7, 6, 2);
	triangles.expandBy(1)->set(7, 2, 3);

	// Top face
	triangles.expandBy(1)->set(1, 5, 4);
	triangles.expandBy(1)->set(1, 4, 0);	
}


int bvCompressedMeshUnitTests()
{
	BvCompressedMeshUnitTests tests;
	tests.runTests();
	return 0;
}


// Test registration
#if defined(HK_COMPILER_MWERKS)
	#pragma fullpath_file on
#endif
HK_TEST_REGISTER(bvCompressedMeshUnitTests , "Fast", "Physics2012/Test/UnitTest/Internal/", __FILE__);

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
