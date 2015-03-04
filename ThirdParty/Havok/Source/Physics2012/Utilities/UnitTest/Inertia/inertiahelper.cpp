/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Physics2012/Collide/Shape/Convex/Box/hkpBoxShape.h>
#include <Physics2012/Collide/Shape/Convex/Sphere/hkpSphereShape.h>
#include <Physics2012/Collide/Shape/Misc/Transform/hkpTransformShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/List/hkpListShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesShape.h>
#include <Common/Internal/ConvexHull/hkGeometryUtility.h>
#include <Common/Base/Math/Matrix/hkMatrix3Util.h>

#include <Physics2012/Dynamics/Entity/hkpRigidBodyCinfo.h>

#include <Physics2012/Utilities/Dynamics/Inertia/hkpInertiaTensorComputer.h>


void HK_CALL getWeirdCompoundTransformShapeMassProperties(hkBool useDuplicates, hkMassProperties& result)
{
	int i;

	// Build a coumpound body.
	// If useDuplicates, fake different mass "primitives" by duplicating 
	// them. eg. addPrim(mass=5) => add 5 prims(mass=1) !

		// These parameters specify the wibbly size. The main (body) sphere has the radius defined
		// below. The arms have size 'boxSize'. 
	hkReal radius = 1.0f;
	hkVector4 boxSize; boxSize.set( 1.0f, 2.0f, 3.0f);


	// We've got 3 shapes (sphere, box, convexvertices)
	int num0 = 5;
	int num1 = 2;
	int num2 = 4;

		// Work out total mass given that each is of uniform unit density, hence has mass = density * volume
		// ie. mass = volume.
		// volume (sphere) = 4/3 PI = 4.18879020478
		// volume (box) = 6
		// volume (thingy) = 7.65268

	hkReal totalMass = hkReal(num0 * 4.18879020478f + num1 * 6 + num2 * 7.65268f);

	
	/////////////////////////////////////////
	///////////// SHAPES ////////////////////
	/////////////////////////////////////////
	hkpSphereShape* sphere = new hkpSphereShape(radius);

	hkVector4 halfExtents; halfExtents.setMul(hkSimdReal_Inv2, boxSize);
	hkpBoxShape* cube1 = new hkpBoxShape(halfExtents, 0 );	// Note: We use HALF-extents for boxes
	cube1->setRadius(0.0f);

	hkArray<hkVector4> vertices;
	//
	// Create the vertices array
	//
	int numSides = 5;
	hkReal radius2 = 1.1f;
	for(i = 0 ; i < numSides; i++)
	{
		hkTransform t;
		t.setIdentity();
		hkVector4 trans; trans.set(radius2, 0, 0);

		hkReal angle = HK_REAL_PI * 2 * i / (hkReal) numSides;
		const hkVector4 axis = hkVector4::getConstant<HK_QUADREAL_0010>();
		hkQuaternion q; q.setAxisAngle(axis, angle);
		trans.setRotatedDir(q, trans);

		hkVector4 v = trans;
		v(2) = -1.0f;
		vertices.pushBack(v);
		v(2) = 1.0f;
		v(0) *= 1.3f;
		v(1) *= 1.3f;
		vertices.pushBack(v);
		
	}
	
	hkpConvexVerticesShape::BuildConfig	config;
	config.m_convexRadius	=	0;
	hkpConvexVerticesShape* shape = new hkpConvexVerticesShape(vertices,config);
	
	/////////////////////////////////////////
	///////////// BODY  ////////////////////
	/////////////////////////////////////////
	hkpRigidBodyCinfo compoundInfo;


	hkArray<hkpShape*> shapeArray;


	
	
	for(i = 0; i < (useDuplicates?num0:1) ; i++)
	{

		hkTransform t;
		hkVector4 v; v.set(-5,1,2);
		v.normalize<3>();
		hkQuaternion r; r.setAxisAngle(v, -0.746f);
		hkVector4 trans; trans.set(1.231f, -0.325f, 0.93252f);
		t.setRotation(r);
		t.setTranslation( trans );
		
		hkpTransformShape* sphereTrans = new hkpTransformShape( sphere, t );

		shapeArray.pushBack(sphereTrans);	
	}
	

	
	for(i = 0; i < (useDuplicates?num1:1); i++)
	{
		hkTransform t;
		hkVector4 v; v.set(3,2,1.1324f);
		v.normalize<3>();
		hkQuaternion r; r.setAxisAngle(v, 0.446f);
		hkVector4 trans; trans.set(.325f, .4363f,-.86427654f);
		t.setRotation(r);
		t.setTranslation( trans );
		
		hkpTransformShape* cube1Trans = new hkpTransformShape( cube1, t );

		shapeArray.pushBack(cube1Trans);	
	}
	
	
	

	for(i = 0; i < (useDuplicates?num2:1); i++)
	{
		hkTransform t;
		hkVector4 v; v.set(.5f, 0.4f, .9f);
		v.normalize<3>();
		hkQuaternion r; r.setAxisAngle(v, 1.2f);
		hkVector4 trans; trans.set(-.435f, -0.5f, -0.2f);
		t.setRotation(r);
		t.setTranslation( trans );

		hkpTransformShape* convexTrans = new hkpTransformShape( shape, t );

		shapeArray.pushBack(convexTrans);	
	}
	
	




	


	// Now we can create the compound body as a hkpListShape
	
	hkpListShape* listShape = new hkpListShape(&shapeArray[0], shapeArray.getSize());
	compoundInfo.m_shape = listShape;

	sphere->removeReference();
	cube1->removeReference();
	shape->removeReference();

	for (i = 0; i < shapeArray.getSize(); ++i)
	{
		shapeArray[i]->removeReference();
	}



	//
	// Create the rigid body 
	//

	compoundInfo.m_mass = totalMass;
		// Fake an inertia tensor using a cube of side 'radius'

	if(useDuplicates)
	{
		hkpInertiaTensorComputer::setShapeVolumeMassProperties(compoundInfo.m_shape, compoundInfo.m_mass, compoundInfo);
	}
	else
	{
		hkArray<hkMassElement> elements;

		// Construct compound inertia tensor from 3 "primitives"
		hkMassElement massElement;
		massElement.m_transform.setIdentity();

			// Actually change order so we DEFINITELY do different calculations than are done for
			// the call above.
		hkpInertiaTensorComputer::computeShapeVolumeMassProperties(listShape->getChildShapeInl(0), num0*4.18879020478f, massElement.m_properties);
		elements.pushBack(massElement);

		hkpInertiaTensorComputer::computeShapeVolumeMassProperties(listShape->getChildShapeInl(1), num1*6.0f, massElement.m_properties);
		elements.pushBack(massElement);

		hkpInertiaTensorComputer::computeShapeVolumeMassProperties(listShape->getChildShapeInl(2), num2*7.65268f, massElement.m_properties);
		elements.pushBack(massElement);

		hkMassProperties massProperties;
		hkpInertiaTensorComputer::combineMassProperties(elements, massProperties);

		compoundInfo.m_centerOfMass = massProperties.m_centerOfMass;
		compoundInfo.m_inertiaTensor = massProperties.m_inertiaTensor;
		compoundInfo.m_mass = massProperties.m_mass;
		
	}

	result.m_centerOfMass = compoundInfo.m_centerOfMass;
	result.m_mass = compoundInfo.m_mass;
	result.m_inertiaTensor =  compoundInfo.m_inertiaTensor;


	listShape->removeReference();
}


	// Create a "box" shape with 8 vertices.
hkpConvexVerticesShape* HK_CALL getBoxConvexVerticesShape(hkVector4 &halfExtents, hkReal radius)
{
	hkArray<hkVector4> vertices;

	vertices.setSize(8);

	vertices[0].set(halfExtents(0), halfExtents(1) ,halfExtents(2));
	vertices[1].set(halfExtents(0), halfExtents(1) ,-halfExtents(2));
	vertices[2].set(halfExtents(0), -halfExtents(1) ,halfExtents(2));
	vertices[3].set(halfExtents(0), -halfExtents(1) ,-halfExtents(2));
	vertices[4].set(-halfExtents(0), halfExtents(1) ,halfExtents(2));
	vertices[5].set(-halfExtents(0), halfExtents(1) ,-halfExtents(2));
	vertices[6].set(-halfExtents(0), -halfExtents(1) ,halfExtents(2));
	vertices[7].set(-halfExtents(0), -halfExtents(1) ,-halfExtents(2));

	hkpConvexVerticesShape::BuildConfig	config;
	config.m_shrinkByConvexRadius	=	false;
	config.m_convexRadius = radius;
	hkpConvexVerticesShape* shape = new hkpConvexVerticesShape(vertices,config);

	return shape;
}

	// Create a "random" set of vertices.
void HK_CALL getRandomConvexVertices(hkArray<hkVector4> &vertices, int numVerts, int seed = 123123)
{
	vertices.setSize(numVerts);

	for(int i = 0; i < numVerts; i++)
	{
		seed = 1664525L*seed+1013904223L;
		hkReal x = (seed & 0x7fff)*(1.0f/(0x7fff));
		seed = 1664525L*seed+1013904223L;
		hkReal y = (seed & 0x7fff)*(1.0f/(0x7fff));
		seed = 1664525L*seed+1013904223L;
		hkReal z = (seed & 0x7fff)*(1.0f/(0x7fff));
		vertices[i].set(x, y ,z);		
	}
}

	// Create a "random" hkpConvexVerticesShape
hkpConvexVerticesShape* HK_CALL getRandomConvexVerticesShape(int numVerts, int seed = 123123)
{
	hkArray<hkVector4> vertices;
	vertices.setSize(numVerts);

	for(int i = 0; i < numVerts; i++)
	{
		seed = 1664525L*seed+1013904223L;
		hkReal x = (seed & 0x7fff)*(1.0f/(0x7fff));
		seed = 1664525L*seed+1013904223L;
		hkReal y = (seed & 0x7fff)*(1.0f/(0x7fff));
		seed = 1664525L*seed+1013904223L;
		hkReal z = (seed & 0x7fff)*(1.0f/(0x7fff));
		vertices[i].set(x, y ,z);		
	}
	
	hkpConvexVerticesShape::BuildConfig	config;
	config.m_shrinkByConvexRadius	=	false;

	hkpConvexVerticesShape* shape = new hkpConvexVerticesShape(vertices,config);

	return shape;

}




void HK_CALL setMassPropertiesToRandomNonsense(hkMassProperties& massProperties)
{
	massProperties.m_centerOfMass.set(-123.123f, 456.456f, 99999.0f);
	massProperties.m_mass = -1.0f;
	massProperties.m_volume = -1.0f;
	hkMatrix3Util::_setDiagonal(-1.0f, -2.0f, -3.0f, massProperties.m_inertiaTensor);
}


hkMatrix3 HK_CALL getBoxInertiaTensor(hkVector4& halfSize, hkReal mass)
{
	hkReal alpha = halfSize(0) * 2.0f;
	hkReal beta = halfSize(1) * 2.0f;
	hkReal gamma = halfSize(2) * 2.0f;

	hkReal ixx = 1.0f / 6.0f;
	hkReal iyy = 1.0f / 6.0f;
	hkReal izz = 1.0f / 6.0f;


	hkReal ixxP = ( (beta*beta) *  (ixx - iyy + izz) * 0.5f )
					+ (   (gamma*gamma) * (ixx + iyy - izz) * 0.5f );

	hkReal iyyP = ( (alpha*alpha) * (-ixx + iyy + izz) * 0.5f)
					+ ( (gamma*gamma) * (ixx + iyy - izz) * 0.5f );

	hkReal izzP = ( (alpha*alpha) * (-ixx + iyy + izz) * 0.5f )
					+ (  (beta*beta) * (ixx - iyy + izz) * 0.5f );

	hkMatrix3 m;
	m.setIdentity();

	m(0,0) = ixxP * mass;
	m(1,1) = iyyP * mass;
	m(2,2) = izzP * mass;
	
	return m;
}

hkQuaternion HK_CALL getQuaternion(hkReal angle, hkReal x, hkReal y, hkReal z)
{
	hkQuaternion q;
	hkVector4 axis; axis.set(x,y,z);
	axis.normalize<3>();
	q.setAxisAngle(axis, angle);
	return q;
}

void HK_CALL transformNasty(hkTransform& t, const hkTransform& nasty)
{
	hkTransform foo;
	foo.setMul(nasty, t);
	t = foo;
}



hkGeometry* HK_CALL getCapsuleGeometry(const hkVector4& start, const hkVector4& end, hkReal radius,  int thetaSamples, int phiSamples, int heightSamples)
{
	hkArray<hkVector4> verts;

		// Create "transform" from start, end.
	hkTransform capsuleToLocal;
	

	hkVector4 axis;
	axis.setSub(end, start);
	hkReal height = axis.length<3>().getReal();
	if(height > 0.0f)
	{
		axis.normalize<3>();

		// find a quat which rotates (1,0,0) to axis
		hkVector4 canonicalZ; canonicalZ.set(0,0,1);
		hkSimdReal axisDot; axisDot.setAbs(axis.dot<3>(canonicalZ));
		if(axisDot < hkSimdReal::fromFloat(1.0f - 1e-5f))
		{
			hkVector4 rotAxis;
			rotAxis.setCross(canonicalZ, axis);
			rotAxis.normalize<3>();

			hkReal rotAngle = hkMath::acos(axis.dot<3>(canonicalZ).getReal());

			hkQuaternion q; q.setAxisAngle(rotAxis, rotAngle);
			capsuleToLocal.setRotation(q);
		}
		else
		{
			capsuleToLocal.setIdentity();	
		}

	}
	else
	{
		capsuleToLocal.setIdentity();
	}

			// Now recenter
	{
		hkVector4 toCentre;
		toCentre.setAdd(start, end);
		toCentre.mul(hkSimdReal_Inv2);
		capsuleToLocal.setTranslation(toCentre);
	}
	
	


	// We'll sweep around the axis of the deflector, from top to bottom, using the original
	// sample directions and data to define the vertices. We'll tessellate in the obvious way.
	// N.B. Top and bottom vertices are added to cap the object. These are calculated as the 
	// average of the surrounding vertices, hence will be slightly flattened.

	int i,j;

	hkVector4 vert;

	hkVector4 bottomInGeom; bottomInGeom.set(0,0,-height/2);
	hkVector4 topInGeom; topInGeom.set(0,0,height/2);
	hkVector4 axisInGeom;
	axisInGeom.setSub(topInGeom, bottomInGeom);
	hkVector4 axisNInGeom; axisNInGeom.set(0,0,1);
	hkVector4 normalInGeom; normalInGeom.set(1,0,0);
	hkVector4 binormalInGeom; binormalInGeom.set(0,-1,0);

	verts.reserveExactly(2 * phiSamples * thetaSamples + 2 + (heightSamples-2) * thetaSamples);

		///////////////////// GET TOP VERTICES ///////////////////
	hkArray<hkVector4> topverts;
	for (i = phiSamples-1 ; i >= 0; i--)
	{
		hkQuaternion qTop; qTop.setAxisAngle(binormalInGeom, hkReal(i) / phiSamples * HK_REAL_PI * .5f);
		hkVector4 topElevation;
		topElevation.setRotatedDir(qTop, normalInGeom);

		hkQuaternion qBottom; qBottom.setAxisAngle(binormalInGeom, -hkReal(i) / phiSamples * HK_REAL_PI * .5f);
		hkVector4 bottomElevation;
		bottomElevation.setRotatedDir(qBottom, normalInGeom);

		for (j = 0; j < thetaSamples; j++)
		{
			hkQuaternion rotationTop; rotationTop.setAxisAngle(axisNInGeom, hkReal(j) / thetaSamples * HK_REAL_PI * 2);			
			hkVector4 topDirection;
			topDirection.setRotatedDir(rotationTop, topElevation);

			hkSimdReal dist; dist.setFromFloat(radius);
			vert.setAddMul(topInGeom, topDirection, dist);

			vert.setTransformedPos(capsuleToLocal, vert);

				// Temporarily store since we'll need these to calculate "top" vertx.
			topverts.pushBack(vert);

		}
	}

	vert.set(0, 0 ,height*0.5f + radius);
	vert.setTransformedPos(capsuleToLocal, vert);

		// Push back top vertex, and then the rest
	verts.pushBack(vert);
	for(i = 0; i < (phiSamples)*thetaSamples; i++)
	{
		verts.pushBack(topverts[i]);
	}

	
		///////////////////// GET MIDDLE VERTICES ///////////////////
	for (j = heightSamples-2; j >= 1; j--)
	{
	
		for (i = 0; i < thetaSamples; i++)
		{	
		//
		// Calculate direction vector for this angle
		//

			hkQuaternion q; q.setAxisAngle(axisNInGeom, hkReal(i) / thetaSamples * HK_REAL_PI * 2);
			hkVector4 direction;
			direction.setRotatedDir(q, normalInGeom);
			
			hkVector4 startx;
			startx.setAddMul(bottomInGeom, axisInGeom, hkSimdReal::fromFloat(hkReal(j) / hkReal(heightSamples - 1)));

			hkSimdReal dist; dist.setFromFloat(radius);

			vert.setAddMul(startx, direction, dist);

			vert.setTransformedPos(capsuleToLocal, vert);

			verts.pushBack(vert);

		}
	}

 


		///////////////////// GET BOTTOM VERTICES ///////////////////
	for (i = 0; i < phiSamples; i++)
	{
		hkQuaternion qTop; qTop.setAxisAngle(binormalInGeom, hkReal(i) / phiSamples * HK_REAL_PI * .5f);
		hkVector4 topElevation;
		topElevation.setRotatedDir(qTop, normalInGeom);

		hkQuaternion qBottom; qBottom.setAxisAngle(binormalInGeom, -hkReal(i) / phiSamples * HK_REAL_PI * .5f);
		hkVector4 bottomElevation;
		bottomElevation.setRotatedDir(qBottom, normalInGeom);

		for (j = 0; j < thetaSamples; j++)
		{
			hkQuaternion rotationBottom; rotationBottom.setAxisAngle(axisNInGeom, hkReal(j) / thetaSamples * HK_REAL_PI * 2);			
			hkVector4 bottomDirection;
			bottomDirection.setRotatedDir(rotationBottom, bottomElevation);

			hkSimdReal dist; dist.setFromFloat(radius);

			vert.setAddMul(bottomInGeom, bottomDirection, dist);
			vert.setTransformedPos(capsuleToLocal, vert);
			verts.pushBack(vert);

		}
	}

	
	vert.set(0, 0 , -(height*0.5f + radius));
	vert.setTransformedPos(capsuleToLocal, vert);

		// Push back bottom vertex
	verts.pushBack(vert);


	
	///////////////////// CONSTRUCT FACE DATA ///////////////////

	hkGeometry* geom = new hkGeometry;
	geom->m_vertices = verts;
	// Right, num samples AROUND axis is thetaSamples.

	// First off, we have thetaSamples worth of faces connected to the top
	int currentBaseIndex = 1;

	hkGeometry::Triangle tr;
	tr.m_material=-1;
	for (i = 0; i < thetaSamples; i++)
	{
		tr.m_a = 0;
		tr.m_b = currentBaseIndex + i;
		tr.m_c = currentBaseIndex + (i+1)%(thetaSamples);
		geom->m_triangles.pushBack(tr);
	}

	// Next we have phi-1 + height-1 + phi-1 lots of thetaSamples*2 worth of faces connected to the previous row
	for(j = 0; j < 2*(phiSamples-1) + heightSamples-1; j++)
	{
		for (i = 0; i < thetaSamples; i++)
		{
			tr.m_a = currentBaseIndex + i;
			tr.m_b = currentBaseIndex + thetaSamples + i;
			tr.m_c = currentBaseIndex + thetaSamples + (i+1)%(thetaSamples);

			geom->m_triangles.pushBack(tr);

			tr.m_b = currentBaseIndex + i;
			tr.m_a = currentBaseIndex + (i+1)%(thetaSamples);
			tr.m_c = currentBaseIndex + thetaSamples + (i+1)%(thetaSamples);
		
			geom->m_triangles.pushBack(tr);
		
		}
		currentBaseIndex += thetaSamples;

	}

	// Finally, we have thetaSamples worth of faces connected to the bottom
	for (i = 0; i < thetaSamples; i++)
	{
		tr.m_b = currentBaseIndex + i;
		tr.m_a = currentBaseIndex + (i+1)%(thetaSamples);
		tr.m_c = currentBaseIndex + thetaSamples;
		geom->m_triangles.pushBack(tr);
	}

	return geom;
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
