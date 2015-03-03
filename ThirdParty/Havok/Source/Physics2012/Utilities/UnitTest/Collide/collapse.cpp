/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>

#include <Physics2012/Utilities/Collide/ShapeUtils/CollapseTransform/hkpTransformCollapseUtil.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Types/Geometry/hkGeometry.h>
#include <Physics2012/Utilities/Dynamics/Inertia/hkpInertiaTensorComputer.h>
#include <Common/Internal/ConvexHull/hkGeometryUtility.h>

/// Dynamics support
#include <Physics2012/Dynamics/Common/hkpProperty.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>

/// Collide support
#include <Physics2012/Collide/Shape/Convex/Box/hkpBoxShape.h>
#include <Physics2012/Collide/Shape/Convex/Capsule/hkpCapsuleShape.h>
#include <Physics2012/Collide/Shape/Convex/Cylinder/hkpCylinderShape.h>
#include <Physics2012/Collide/Shape/Convex/Sphere/hkpSphereShape.h>
#include <Physics2012/Collide/Shape/Misc/Transform/hkpTransformShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/List/hkpListShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexTransform/hkpConvexTransformShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexTranslate/hkpConvexTranslateShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/ExtendedMeshShape/hkpExtendedMeshShape.h>


// test of the hkCollapseTransforms utility (used by the BakeTransformShapes filter)
 

//create a placeholder rigid body
static hkpRigidBody* createRB( const hkpShape* shape )
{
	hkpRigidBodyCinfo info;
	{
		info.m_shape = shape;

		info.m_position.set( hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01() );			
		hkVector4 axis; axis.set( 0.0f, 1.0f, 0.0f );
		info.m_rotation.setAxisAngle( axis, HK_REAL_PI / ((hkUnitTest::rand01() * 4.0f) + 0.01f) );
		info.m_linearVelocity.set( hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01() );
		info.m_angularVelocity.set( hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01() );

		// [1,1,1] vector has length sqrt(3) not 1
		info.m_maxAngularVelocity = hkUnitTest::randRange( 1.733f, 100.0f );
		info.m_maxLinearVelocity = hkUnitTest::randRange( 1.733f, 100.0f );
		info.m_linearDamping = hkUnitTest::rand01();
		info.m_angularDamping = hkUnitTest::rand01();

		info.m_friction = hkUnitTest::rand01();
		info.m_restitution = hkUnitTest::rand01();			
		info.m_motionType = hkpMotion::MOTION_DYNAMIC;							
		info.m_qualityType = HK_COLLIDABLE_QUALITY_FIXED;
	}

	hkpRigidBody* body = new hkpRigidBody( info );
	return body;
}
 
//create a placeholder convex vertices shape
static hkpConvexVerticesShape* createCVS()
{
	int numVertices = 4;
	int stride = 16;
	
	HK_ALIGN_REAL(hkReal vertices[]) = { // 4 vertices plus padding
		-2.0f, 2.0f, 1.0f, 0.0f, // v0
		 1.0f, 3.0f, 0.0f, 0.0f, // v1
		 0.0f, 1.0f, 3.0f, 0.0f, // v2
		 1.0f, 0.0f, 0.0f, 0.0f  // v3
	};

	hkStridedVertices stridedVerts;
	{
		stridedVerts.m_numVertices = numVertices;
		stridedVerts.m_striding = stride;
		stridedVerts.m_vertices = vertices;
	}
	return new hkpConvexVerticesShape(stridedVerts);
}

// definitions for the hkpMeshShape
//
const int NUM_VERTICES = 4;
const int NUM_TRIANGLES = 2;
const int NUM_DEGENERATE_TRIANGLES = 2;

//create a mesh shape
static hkpExtendedMeshShape* createMesh( hkVector4* vertices, hkUint16 numVertices, hkUint16* indices, hkUint16 numIndices )
{
	// create vertices
	vertices[0].set( 1.0f, 0.0f, 1.0f );
	vertices[1].set(-1.0f, 0.0f, 1.0f );
	vertices[2].set( 1.0f, 0.0f,-1.0f );
	vertices[3].set(-1.0f, 0.0f,-1.0f );

	// create the first non-degenerate triangle (0,1,2)
	indices[0] = 0;	
	indices[1] = 1;	
	indices[2] = 2;	

	// create a degenerate triangle (1,2,2)
	indices[3] = 2;	

	// create a degenerate triangle (2,2,1)
	indices[4] = 1;

	// create the second non-degenerate triangle (2,1,3)
	indices[5] = 3;	

	// create shapes
	hkpExtendedMeshShape* meshShape = new hkpExtendedMeshShape();
	{
		hkpExtendedMeshShape::TrianglesSubpart part;

		part.m_vertexBase = &(vertices[0](0));
		part.m_vertexStriding = sizeof(hkVector4);
		part.m_numVertices = NUM_VERTICES;

		part.m_indexBase = indices;
		part.m_indexStriding = sizeof( hkUint16 );
		part.m_numTriangleShapes = NUM_TRIANGLES + NUM_DEGENERATE_TRIANGLES;
		part.m_stridingType = hkpExtendedMeshShape::INDICES_INT16;

		hkVector4 s; s.set(20.0f, 20.0f, 20.0f);
		part.setScaling( s );

		meshShape->addTrianglesSubpart( part );
	}

	return meshShape;	
}

//define various test cases.

enum transformType
{
	identityTT,    //identity
	translationTT, //translation
	fullTT		   //translation+rotation
};
struct collapseTestCase
{
	hkpShapeType   inputTST;  // the input hkpTransformShape type
	transformType inputTT;   // the input hkpTransformShape transform type
	hkpShapeType   inputST;   // the input hkpTransformShape shape type	 
	hkpShapeType   outputTST; // the (expected) output shape type 
	hkpShapeType   outputST;  // the output shape's (expected) child shape type, if there is a child (otherwise hkcdShapeType::INVALID) 
	                         //(there is a child if outputTST is an hkpTransformShape, hkpConvexTransformShape, or hkpConvexTranslateShape)
};
static int numtestcases = 12;
static const collapseTestCase g_testVariants[] =
{
//																				(expected)					(expected)
//    input shape type:			  input transform:  input shape:				output shape type:			output shape child:
	{ hkcdShapeType::CONVEX_TRANSFORM,  identityTT,		hkcdShapeType::BOX,				 hkcdShapeType::BOX,					hkcdShapeType::INVALID },
	{ hkcdShapeType::CONVEX_TRANSFORM,  translationTT,   hkcdShapeType::BOX,				 hkcdShapeType::CONVEX_TRANSLATE,	hkcdShapeType::BOX     },
	{ hkcdShapeType::CONVEX_TRANSFORM,  fullTT,			hkcdShapeType::BOX,				 hkcdShapeType::CONVEX_TRANSFORM,	hkcdShapeType::BOX     },
	{ hkcdShapeType::CONVEX_TRANSFORM,  fullTT,		    hkcdShapeType::CAPSULE,			 hkcdShapeType::CAPSULE,				hkcdShapeType::INVALID },
	{ hkcdShapeType::CONVEX_TRANSFORM,  identityTT,		hkcdShapeType::CONVEX_VERTICES,   hkcdShapeType::CONVEX_VERTICES,		hkcdShapeType::INVALID },
	{ hkcdShapeType::CONVEX_TRANSFORM,  fullTT,		    hkcdShapeType::CONVEX_VERTICES,   hkcdShapeType::CONVEX_VERTICES,		hkcdShapeType::INVALID },
	
	{ hkcdShapeType::CONVEX_TRANSLATE,  identityTT,		hkcdShapeType::BOX,				 hkcdShapeType::BOX,					hkcdShapeType::INVALID },
	{ hkcdShapeType::CONVEX_TRANSLATE,  translationTT,   hkcdShapeType::BOX,				 hkcdShapeType::CONVEX_TRANSLATE,	hkcdShapeType::BOX     },
	{ hkcdShapeType::CONVEX_TRANSLATE,  fullTT,			hkcdShapeType::BOX,				 hkcdShapeType::CONVEX_TRANSLATE,	hkcdShapeType::BOX     },
	{ hkcdShapeType::CONVEX_TRANSLATE,  fullTT,		    hkcdShapeType::CAPSULE,			 hkcdShapeType::CAPSULE,				hkcdShapeType::INVALID },
	{ hkcdShapeType::CONVEX_TRANSLATE,  identityTT,		hkcdShapeType::CONVEX_VERTICES,   hkcdShapeType::CONVEX_VERTICES,		hkcdShapeType::INVALID },
	{ hkcdShapeType::CONVEX_TRANSLATE,  fullTT,		    hkcdShapeType::CONVEX_VERTICES,   hkcdShapeType::CONVEX_VERTICES,		hkcdShapeType::INVALID }
};


// check that various shapes wrapped in an hkpTransformShape, hkpConvexTransformShape, or hkpConvexTranslateShape are correctly collapsed
static void checkCollapse()
{			
	hkVector4 rvec1; rvec1.set( hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01());
	hkVector4 rvec2; rvec2.set( hkUnitTest::rand01(), hkUnitTest::rand01(), hkUnitTest::rand01());

	hkpBoxShape* boxShape = new hkpBoxShape( rvec1 );
	hkpCapsuleShape* capsuleShape = new hkpCapsuleShape( rvec1, rvec2, hkUnitTest::rand01() );
	hkpConvexVerticesShape* convexVerticesShape = createCVS();
	
	hkTransform in_t;  //the input transform
	hkVector4 t_vec; t_vec.set(1, 0, 0);
	hkQuaternion t_quat; t_quat.setIdentity(); t_quat.setAxisAngle(t_vec, hkReal(0.25) * HK_REAL_PI);

	hkpConvexShape* in_s = HK_NULL; //the input (convex) shape
	hkpRigidBody* body = HK_NULL;

	const hkpShape* out_s; //the output shape
	hkpShapeType outTST;   //the output shape type
	hkpShapeType outST;    //the output shape's child's type

	for (int n=0; n<numtestcases; n++)
	{
		//input transform
		switch ( g_testVariants[n].inputTT )
		{
			case identityTT:
				{
					in_t.setIdentity();
					break;
				}
			case translationTT:
				{
					in_t.setIdentity();
					in_t.setTranslation(t_vec);
					break;
				}
			case fullTT:
				{
					in_t.setIdentity();
					in_t.setTranslation(t_vec);
					in_t.setRotation(t_quat);
					break;
				}
		}

		//input shape
		switch ( g_testVariants[n].inputST )
		{
			case hkcdShapeType::BOX:
				in_s = boxShape;
				break;

			case hkcdShapeType::CAPSULE:
				in_s = capsuleShape;
				break;

			case hkcdShapeType::CONVEX_VERTICES:
				in_s = convexVerticesShape;
				break;

			default:
				break;
		}

		//create a rigid body with the input transform and shape hierarchy, and apply collapse()
		switch ( g_testVariants[n].inputTST )
		{
			case hkcdShapeType::CONVEX_TRANSFORM:
				{
					hkpConvexTransformShape* ts = new hkpConvexTransformShape( in_s, in_t );
					body = createRB( ts );
					ts->removeReference();
					break;
				}
			case hkcdShapeType::CONVEX_TRANSLATE:
				{
					hkpConvexTranslateShape* ts = new hkpConvexTranslateShape( in_s, in_t.getTranslation() );
					body = createRB( ts );
					ts->removeReference();
					break;
				}
			default:
				break;
		}

		hkpTransformCollapseUtil::Options options;
		hkpTransformCollapseUtil::Results results;
		hkpTransformCollapseUtil::collapseTransforms(body, options, results);

		//check the output shape hierarchy
		if (body)
		{
			out_s = body->getCollidable()->getShape();
			outTST = out_s->getType();
			
			switch ( outTST )
			{
				case hkcdShapeType::TRANSFORM:
					{
						const hkpTransformShape* tforms = static_cast<const hkpTransformShape*> (out_s);
						outST = tforms->getChildShape()->getType();
						break;
					}
				case hkcdShapeType::CONVEX_TRANSFORM:
					{
						const hkpConvexTransformShape* ctforms = static_cast<const hkpConvexTransformShape*> (out_s);
						outST = ctforms->getChildShape()->getType();
						break;
					}	
				case hkcdShapeType::CONVEX_TRANSLATE:
					{
						const hkpConvexTranslateShape* ctlates = static_cast<const hkpConvexTranslateShape*> (out_s);
						outST = ctlates->getChildShape()->getType();
						break;
					}
				default:
					{
						outST = hkcdShapeType::INVALID;
						break;
					}
			}
			
			HK_TEST( outTST == g_testVariants[n].outputTST );
			HK_TEST( outST  == g_testVariants[n].outputST );

			body->removeReference();
		}
	}

	boxShape->removeReference();
	capsuleShape->removeReference();
	convexVerticesShape->removeReference();

}


// check that shapes wrapped in an hkpListShape are correctly collapsed
static void checkListCollapse()
{	
	hkTransform in_t;  //the input transform
	hkVector4 t_vec; t_vec.set(1, 0, 0);
	hkQuaternion t_quat; t_quat.setIdentity(); t_quat.setAxisAngle(t_vec, hkReal(0.25) * HK_REAL_PI);

	in_t.setIdentity();
	in_t.setTranslation(t_vec);
	in_t.setRotation(t_quat);

	int nChildren = 5;
	hkpShape** listShapes = hkAllocate<hkpShape*>(nChildren, HK_MEMORY_CLASS_DEMO);

	//create a transform shape ts pointing to a convex vertices shape
	hkpConvexVerticesShape* in_s = createCVS();
	hkpTransformShape* ts = new hkpTransformShape( in_s, in_t );
	in_s->removeReference();

	//create a list shape with several child shapes all pointing to ts
	for (int n=0; n<nChildren; n++) 
	{	
		listShapes[n] = ts;			
	}
	hkpListShape* ls = new hkpListShape(listShapes, nChildren);
	ts->removeReference();
	
	hkDeallocate( listShapes );
	hkpRigidBody* body = createRB( ls );
	ls->removeReference();	

	//do the collapse
	hkpTransformCollapseUtil::Options options;
	hkpTransformCollapseUtil::Results results;
	hkpTransformCollapseUtil::collapseTransforms(body, options, results);

	//see that the output shape is a list
	const hkpShape* out_s = body->getCollidable()->getShape();
	hkpShapeType outTST = out_s->getType();
	HK_TEST( outTST == hkcdShapeType::LIST );
	if ( outTST != hkcdShapeType::LIST ) return;
	
	//see that the output list child shapes are now all convex vertices shapes rather than transform shapes
	const hkpListShape* check_ls = static_cast<const hkpListShape*> (out_s);
	for (int n=0; n<nChildren; n++) 
	{
		const hkpShape* cs = check_ls->getChildShapeInl(n);
		hkpShapeType csType = cs->getType();
		HK_TEST( csType == hkcdShapeType::CONVEX_VERTICES );
	}
	
	body->removeReference();
	
}


//check that a list shape whose children are different nested transform shapes is correctly collapsed
/*

		RB - LS            =>       RB - LS'
			/ | \                       / | \
		   /  |  \                     /  |  \
		  /   |   \                   /   |   \
		 T1   T2  T3               CVS1' CVS2' CVS3'
         |    |    |
	     .    .    .
		 .    .    .
		 |    |    |
	    CVS1 CVS2 CVS3

*/
static void checkListOfDifferentNestedTransformCollapse()
{
	//initialize transform shapes
	hkVector4 t_vec; t_vec.set(1, 0, 0);
	hkQuaternion t_quat; t_quat.setIdentity(); t_quat.setAxisAngle(t_vec, hkReal(0.25) * HK_REAL_PI);
	hkTransform in_t;
	in_t.setIdentity();
	in_t.setTranslation(t_vec);
	in_t.setRotation(t_quat);

	//create a list shape with nChildren different child shapes
	int nChildren = 5;
	hkpShape** listShapes = hkAllocate<hkpShape*> (nChildren, HK_MEMORY_CLASS_DEMO);

	for (int n=0; n<nChildren; n++) 
	{	
		//each child shape is a chain of mTransformShapes hkTransformShapes (followed by a convex vertices shape)
		int mTransformShapes = 3;

		hkpConvexVerticesShape* cvs = createCVS();
		hkpTransformShape* ts_prev = new hkpTransformShape( cvs, in_t );
		cvs->removeReference();

		hkpTransformShape *ts_next = ts_prev;
		for (int m=0; m<mTransformShapes-1; m++)
		{
			ts_next = new hkpTransformShape( ts_prev, in_t );
			ts_prev->removeReference();
			ts_prev = ts_next;
		}
		listShapes[n] = ts_next;			
	}
	
	hkpListShape* ls = new hkpListShape(listShapes, nChildren);
	
	for (int n=0; n<nChildren; n++) 
	{
		const hkpShape* childShape = ls->getChildShapeInl(n);
		childShape->removeReference();	
	}

	hkDeallocate( listShapes );
	hkpRigidBody* body = createRB( ls );
	ls->removeReference();

	//do the collapse
	hkpTransformCollapseUtil::Options options;
	hkpTransformCollapseUtil::Results results;
	hkpTransformCollapseUtil::collapseTransforms(body, options, results);

	//see that the output shape is a list
	const hkpShape* out_s = body->getCollidable()->getShape();
	hkpShapeType outTST = out_s->getType();
	HK_TEST( outTST == hkcdShapeType::LIST );
	if ( outTST != hkcdShapeType::LIST ) return;
	
	//see that the output list child shapes are now all convex vertices shapes rather than transform shapes
	const hkpListShape* check_ls = static_cast<const hkpListShape*> (out_s);
	for (int n=0; n<nChildren; n++) 
	{
		const hkpShape* cs = check_ls->getChildShapeInl(n);
		hkpShapeType csType = cs->getType();
		HK_TEST( csType == hkcdShapeType::CONVEX_VERTICES );
	}

	body->removeReference();
}

//check that a list shape whose children are the same nested transform shape is correctly collapsed
/*

	RB - LS                 =>      RB - LS'
		/ | \                           / | \
	   /  |  \                         /  |  \
	   \  |  /                         \  |  /
	    \ | /                           \ | /
		  T - T - CVS                    CVS'

*/
static void checkListOfNestedTransformCollapse()
{
	//initialize transform shapes
	hkVector4 t_vec; t_vec.set(1, 0, 0);
	hkQuaternion t_quat; t_quat.setIdentity(); t_quat.setAxisAngle(t_vec, hkReal(0.25) * HK_REAL_PI);
	hkTransform in_t;
	in_t.setIdentity();
	in_t.setTranslation(t_vec);
	in_t.setRotation(t_quat);

	hkpConvexVerticesShape* cvs = createCVS();
	hkpTransformShape* ts1 = new hkpTransformShape( cvs, in_t );
	cvs->removeReference();
	hkpTransformShape* ts2 = new hkpTransformShape( ts1, in_t );
	ts1->removeReference();
	hkpTransformShape* ts3 = new hkpTransformShape( ts2, in_t );
	ts2->removeReference();

	int nChildren = 5;
	hkpShape** listShapes = hkAllocate<hkpShape*>( nChildren, HK_MEMORY_CLASS_DEMO );

	//create a list shape with several child shapes all pointing to ts3
	for (int n=0; n<nChildren; n++) 
	{	
		listShapes[n] = ts3;			
	}
	hkpListShape* ls = new hkpListShape(listShapes, nChildren);
	ts3->removeReference();

	hkDeallocate( listShapes );
	hkpRigidBody* body = createRB( ls );
	ls->removeReference();

	//do the collapse
	hkpTransformCollapseUtil::Options options;
	hkpTransformCollapseUtil::Results results;
	hkpTransformCollapseUtil::collapseTransforms(body, options, results);

	//see that the output shape is a list
	const hkpShape* out_s = body->getCollidable()->getShape();
	hkpShapeType outTST = out_s->getType();
	HK_TEST( outTST == hkcdShapeType::LIST );
	if ( outTST != hkcdShapeType::LIST ) return;
	
	//see that the output list child shapes are now all -- the same -- convex vertices shape
	//(testing the hkShapeReplacementMap)
	const hkpListShape* check_ls = static_cast<const hkpListShape*> (out_s);
	
	hkpShape* cs = const_cast<hkpShape*>(check_ls->getChildShapeInl(0));
	hkpShape* cs_prev = cs;
	bool allSame = true;
	for (int n=1; n<nChildren; n++) 
	{
		cs = const_cast<hkpShape*>(check_ls->getChildShapeInl(n));

		hkpShapeType csType = cs->getType();
		HK_TEST( csType == hkcdShapeType::CONVEX_VERTICES );

		if (cs != cs_prev) 
		{
			allSame = false;
			HK_TEST( allSame );
			break;
		}
		cs_prev = cs;
	}

	body->removeReference();
}

//check that a list shape whose children are also list shapes is correctly collapsed
static void checkListOfListCollapse()
{
	//initialize transform shapes
	hkVector4 t_vec; t_vec.set(1, 0, 0);
	hkQuaternion t_quat; t_quat.setIdentity(); t_quat.setAxisAngle(t_vec, hkReal(0.25) * HK_REAL_PI);
	hkTransform in_t;
	in_t.setIdentity();
	in_t.setTranslation(t_vec);
	in_t.setRotation(t_quat);

	//create a list shape with nChildren different child shapes
	int nChildren = 5;
	hkpShape** listShapes = hkAllocate<hkpShape*> (nChildren, HK_MEMORY_CLASS_DEMO);

	for (int n=0; n<nChildren; n++) 
	{	
		//each child shape is a list shape with mTransformShapes children (each a transform shape followed by a convex vertices shape)
	
		int mTransformShapes = 3;
		hkpShape** childlistShapes = hkAllocate<hkpShape*> (mTransformShapes, HK_MEMORY_CLASS_DEMO);

		for (int m=0; m<mTransformShapes; m++)
		{
			hkpConvexVerticesShape* cvs = createCVS();
			hkpTransformShape* ts = new hkpTransformShape( cvs, in_t );
			cvs->removeReference();
			childlistShapes[m] = ts;
		}
				
		hkpListShape* cls = new hkpListShape(childlistShapes, mTransformShapes);
		listShapes[n] = cls;

		for (int m=0; m<mTransformShapes; m++)
		{
			const hkpShape* childlistShape = cls->getChildShapeInl(m);
			childlistShape->removeReference();	
		}

		hkDeallocate( childlistShapes );
	}

	hkpListShape* ls = new hkpListShape(listShapes, nChildren);
		
	for (int n=0; n<nChildren; n++) 
	{
		const hkpShape* childShape = ls->getChildShapeInl(n);
		childShape->removeReference();	
	}
	
	hkDeallocate( listShapes );
	hkpRigidBody* body = createRB( ls );
	ls->removeReference();

	//do the collapse
	hkpTransformCollapseUtil::Options options;
	hkpTransformCollapseUtil::Results results;
	hkpTransformCollapseUtil::collapseTransforms(body, options, results);

	//see that the output shape is a list
	const hkpShape* out_s = body->getCollidable()->getShape();
	hkpShapeType outTST = out_s->getType();
	HK_TEST( outTST == hkcdShapeType::LIST );
	if ( outTST != hkcdShapeType::LIST ) return;

	//see that the output list child shapes are list shapes whose children are convex vertices shapes
	const hkpListShape* check_ls = static_cast<const hkpListShape*> (out_s);
	for (int n=0; n<nChildren; n++) 
	{
		const hkpShape* cs = check_ls->getChildShapeInl(n);
		hkpShapeType csType = cs->getType();
		HK_TEST( csType == hkcdShapeType::LIST );

		const hkpListShape* check_cls = static_cast<const hkpListShape*> (cs);
		int mChildren = check_cls->getNumChildShapes();
		for (int m=0; m<mChildren; m++)
		{
			const hkpShape* cls = check_cls->getChildShapeInl(m);
			hkpShapeType clsType = cls->getType();
			HK_TEST( clsType == hkcdShapeType::CONVEX_VERTICES );
		}
	}

	body->removeReference();
}


//check that a more complicated situation is correctly collapsed
/*

	RB - LS                 =>      RB - LS'
		/ | \                           / | \
       /  |  \                         /  |  \
       \  |   \                        \  |   \
	    \ |    \                        \ |    \
		 T1    T2                       CVS'   CVS''
		  \    |
		   \   |
		    \  |
			  T - CVS

*/
static void checkOddCase()
{
	//initialize transform shapes
	hkVector4 t_vec; t_vec.set(1, 0, 0);
	hkQuaternion t_quat; t_quat.setIdentity(); t_quat.setAxisAngle(t_vec, hkReal(0.25) * HK_REAL_PI);
	hkTransform in_t;
	in_t.setIdentity();
	in_t.setTranslation(t_vec);
	in_t.setRotation(t_quat);

	hkpConvexVerticesShape* cvs = createCVS();
	hkpTransformShape* ts = new hkpTransformShape( cvs, in_t );
	cvs->removeReference();

	hkpTransformShape* ts1 = new hkpTransformShape( ts, in_t );
	hkpTransformShape* ts2 = new hkpTransformShape( ts, in_t );
	ts->removeReference();

	//create a list shape with 3 different children
	hkpShape** listShapes = hkAllocate<hkpShape*>(3, HK_MEMORY_CLASS_DEMO);

	//point first two children to ts1
	for (int n=0; n<2; n++) 
	{	
		listShapes[n] = ts1;			
	}
	//point last child to ts2
	listShapes[2] = ts2;
	
	hkpListShape* ls = new hkpListShape(listShapes, 3);
	ts1->removeReference();
	ts2->removeReference();
	
	hkDeallocate( listShapes );
	hkpRigidBody* body = createRB( ls );
	ls->removeReference();

	//do the collapse
	hkpTransformCollapseUtil::Options options;
	hkpTransformCollapseUtil::Results results;
	hkpTransformCollapseUtil::collapseTransforms(body, options, results);

	//see that the output shape is a list
	const hkpShape* out_s = body->getCollidable()->getShape();
	hkpShapeType outTST = out_s->getType();
	HK_TEST( outTST == hkcdShapeType::LIST );
	if ( outTST != hkcdShapeType::LIST ) return;

	//check the first two child shapes are the same convex vertices shape
	const hkpListShape* check_ls = static_cast<const hkpListShape*> (out_s);

	hkpShape* cs0 = const_cast<hkpShape*>(check_ls->getChildShapeInl(0));
	hkpShapeType cs0Type = cs0->getType();
	HK_TEST( cs0Type == hkcdShapeType::CONVEX_VERTICES );
	
	hkpShape* cs1 = const_cast<hkpShape*>(check_ls->getChildShapeInl(1));
	HK_TEST( cs1 == cs0 );

	//check that the last child shape is a (different) convex vertices shape
	hkpShape* cs2 = const_cast<hkpShape*>(check_ls->getChildShapeInl(2));
	hkpShapeType cs2Type = cs2->getType();
	HK_TEST( cs2Type == hkcdShapeType::CONVEX_VERTICES );
	HK_TEST ( cs2 != cs0 );

	body->removeReference();

}

 

/////////////////////////////////////////////////////////////////////////////////////
static int collapse_main()
{
	hkError::getInstance().setEnabled(0x2ff8c16f, false); //disable an annoying warning

	checkCollapse();
	checkListCollapse();
	checkListOfDifferentNestedTransformCollapse();
	checkListOfNestedTransformCollapse();
	checkListOfListCollapse();
	checkOddCase();

	hkError::getInstance().setEnabled(0x2ff8c16f, true);

	return 0;

}   

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(collapse_main, "Slow", "Physics2012/Test/UnitTest/Utilities/", __FILE__     );

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
