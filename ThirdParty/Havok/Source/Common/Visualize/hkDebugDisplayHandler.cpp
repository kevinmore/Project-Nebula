/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Visualize/hkVisualize.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>
#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>
#include <Common/Base/Types/Geometry/Aabb/hkAabb.h>
#include <Common/Base/Types/Color/hkColor.h>
#include <Common/Visualize/hkDebugDisplayHandler.h>
#include <Common/Visualize/hkDisplayGeometryBuilder.h>
#include <Common/Visualize/Shape/hkDisplayGeometry.h>


hkResult hkDebugDisplayHandler::addGeometry(hkDisplayGeometry* geometry, hkUlong id, int tag, hkUlong shapeIdHint)
{
	hkInplaceArray<hkDisplayGeometry*, 1> arrayOfGeometries;
	arrayOfGeometries.pushBackUnchecked( geometry );
	return addGeometry(arrayOfGeometries, geometry->getTransform(), id, tag, shapeIdHint);
}

hkResult hkDebugDisplayHandler::updateGeometry( const hkQsTransform& transform, hkUlong id, int tag )
{
	hkMatrix4 transformAsMatrix;
	transformAsMatrix.set( transform );

	return updateGeometry(transformAsMatrix, id, tag);
}

hkResult hkDebugDisplayHandler::skinGeometry(hkUlong* ids, int numIds, const hkQsTransform* poseModel, int numPoseModel, const hkQsTransform& worldFromModel, int tag )
{
	hkLocalBuffer<hkMatrix4> poseWorldAsMatrices(numPoseModel);
	for ( hkInt32 i = 0; i < numPoseModel; ++i )
	{
		poseWorldAsMatrices[i].set(poseModel[i]);
	}

	hkMatrix4 worldFromModelAsMatrix;
	worldFromModelAsMatrix.set(worldFromModel);

	return skinGeometry(ids, numIds, poseWorldAsMatrices.begin(), numPoseModel, worldFromModelAsMatrix, tag);
}


void hkDebugDisplayHandler::displayFrame( const hkQsTransform& worldFromLocal, hkReal size, int id, int tag )
{
	hkVector4 ZERO;
	hkVector4 X;
	hkVector4 Y;
	hkVector4 Z;

	hkVector4 vec; vec.setZero();
	ZERO.setTransformedPos( worldFromLocal, vec );
	vec.set( size, 0, 0, 0 );
	X.setTransformedPos( worldFromLocal, vec );
	vec.set( 0, size, 0, 0 );
	Y.setTransformedPos( worldFromLocal, vec );
	vec.set( 0, 0, size, 0 );
	Z.setTransformedPos( worldFromLocal, vec );

	hkVector4 dirX; dirX.setSub( X, ZERO );
	hkVector4 dirY; dirY.setSub( Y, ZERO );
	hkVector4 dirZ; dirZ.setSub( Z, ZERO );

	displayArrow( ZERO, dirX, hkColor::RED, id, tag );
	displayArrow( ZERO, dirY, hkColor::GREEN, id, tag );
	displayArrow( ZERO, dirZ, hkColor::BLUE, id, tag );
}

void hkDebugDisplayHandler::displayFrame( const hkTransform& worldFromLocal, hkReal size, int id, int tag )
{
	hkQsTransform t;
	t.setFromTransform( worldFromLocal );
	displayFrame( t, size, id, tag );
}

void hkDebugDisplayHandler::displayArrow(const hkVector4& from, const hkVector4& dir, hkColor::Argb color, int id, int tag)
{
	// Check that we have a valid direction
	if (dir.lengthSquared<3>().getReal() < HK_REAL_EPSILON)
	{
		return;
	}

	hkVector4 to; to.setAdd( from, dir );
	hkVector4 ort; hkVector4Util::calculatePerpendicularVector( dir, ort );
	ort.normalize<3>();
	hkVector4 ort2; ort2.setCross( dir, ort );

	ort.mul( dir.length<3>() );

	hkSimdReal c0; c0.setFromFloat(0.85f);
	hkSimdReal c; c.setFromFloat(1.0f - 0.85f);
	hkVector4 p; p.setInterpolate( from, to, c0 );
	hkVector4 p0; p0.setAddMul( p, ort, c );
	hkVector4 p1; p1.setAddMul( p, ort, -c );
	hkVector4 p2; p2.setAddMul( p, ort2, c );
	hkVector4 p3; p3.setAddMul( p, ort2, -c );

	displayLine( from, to, color, id, tag );
	displayLine( to, p0, color, id, tag );
	displayLine( to, p1, color, id, tag );
	displayLine( to, p2, color, id, tag );
	displayLine( to, p3, color, id, tag );
}

void hkDebugDisplayHandler::displayStar(const hkVector4& position, hkReal scale, hkColor::Argb color, int id, int tag)
{
	for (int k=0; k<3; k++)
	{
		hkVector4 star, pt1, pt2;
		star.setZero();

		star(k) = scale;
		pt1.setAdd(position,star);
		pt2.setSub(position,star);
		displayLine(pt1, pt2, color, id, tag);
	}
}

void hkDebugDisplayHandler::displayModelSpacePose(int numTransforms, const hkInt16* parentIndices, const hkQsTransform* modelSpacePose, const hkQsTransform& worldFromModel, hkColor::Argb color, int id, int tag)
{
	const hkInt16 numBones = static_cast<hkInt16> (numTransforms);

	for (hkInt16 i = 0; i < numBones; i++)
	{
		hkVector4 p1;

		p1.setTransformedPos(worldFromModel, modelSpacePose[i].getTranslation());

		hkReal boneLen = 1.0f;

		const hkInt16 parent = parentIndices[i];

		// Display connections between child and parent
		if (parent == -1)
		{
			hkVector4 p2;
			p2.set(0,0,0);

			p2.setTransformedPos(worldFromModel, p2);
			displayLine(p1, p2, color, id, tag);			
		}
		else
		{
			hkVector4 p2;
			p2.setTransformedPos(worldFromModel, modelSpacePose[parent].getTranslation());
			displayLine(p1, p2, color, id, tag);

			hkVector4 bone; bone.setSub(p1,p2);
			boneLen = bone.length<3>().getReal();
			boneLen = (boneLen > 10.0f) ? 10.0f : boneLen;
			boneLen = (boneLen < 0.1f) ? 0.1f : boneLen;
		}


		const hkVector4 worldPosition = p1;
		hkQuaternion worldRotation; worldRotation.setMul(worldFromModel.getRotation(), modelSpacePose[i].getRotation());

		// Display local axis 
		{ 
			hkVector4 boneLocalFrame;			
			const hkReal boneAxisSize = 0.25f;

			hkVector4 p2;
			p1 = worldPosition;			
			boneLocalFrame.set(boneAxisSize * boneLen, 0, 0, 0);
			p2.setRotatedDir(worldRotation, boneLocalFrame);
			p2.add(p1);
			displayLine(p1, p2, 0x7fff0000, id, tag);
			p1 = worldPosition;
			boneLocalFrame.set(0, boneAxisSize * boneLen, 0);
			p2.setRotatedDir( worldRotation, boneLocalFrame);
			p2.add(p1);
			displayLine(p1, p2, 0x7f00ff00, id, tag);
			p1 = worldPosition;
			boneLocalFrame.set(0, 0, boneAxisSize * boneLen);
			p2.setRotatedDir(worldRotation, boneLocalFrame);
			p2.add(p1);
			displayLine(p1, p2, 0x7f0000ff, id, tag);
		}		
	}
}

void hkDebugDisplayHandler::displayLocalSpacePose(int numTransforms, const hkInt16* parentIndices, const hkQsTransform* localSpacePose, const hkQsTransform& worldFromModel, hkColor::Argb color, int id, int tag)
{
	// Transform to model space (from hkaSkeletonUtils::transformLocalPoseToModelPose)
	hkLocalBuffer<hkQsTransform> modelSpacePose(numTransforms);
	{
		for( int i = 0; i < numTransforms; i++ )
		{
			const int parentId = parentIndices[i];
			HK_ASSERT(0x675efb2b,  (parentIndices[i] == -1) || (parentIndices[i] < i) );
			const hkQsTransform& worldFromParent = ( -1 != parentId ) ? modelSpacePose[parentId] : hkQsTransform::getIdentity();
			modelSpacePose[i].setMul( worldFromParent, localSpacePose[i] );
		}
	}

	displayModelSpacePose(numTransforms, parentIndices, modelSpacePose.begin(), worldFromModel, color, id, tag);
}

hkResult hkDebugDisplayHandler::addGeometryLazily( const hkReferencedObject* source, hkDisplayGeometryBuilder* builder, const hkTransform& transform, hkUlong id, int tag, hkUlong shapeIdHint )
{
	hkInplaceArray<hkDisplayGeometry*,8> displayGeometries;

	builder->buildDisplayGeometries( source, displayGeometries );
	
	for(int i = (displayGeometries.getSize() - 1); i >= 0; i--)
	{
		if( ( displayGeometries[i]->getType() == HK_DISPLAY_CONVEX ) &&
			( displayGeometries[i]->getGeometry() == HK_NULL ) )
		{
			HK_REPORT( "Unable to build display geometry from source" );
			displayGeometries[i]->removeReference();
			displayGeometries.removeAt( i );
		}
	}

	hkResult result = HK_FAILURE;
	
	if ( displayGeometries.getSize() > 0 )
	{
		result = addGeometry( displayGeometries, transform, id, tag, shapeIdHint );
	}
	
	hkReferencedObject::removeReferences( displayGeometries.begin(), displayGeometries.getSize() );

	return result;
}

hkResult hkDebugDisplayHandler::addGeometryHash( const hkReferencedObject* source, hkDisplayGeometryBuilder* builder, const Hash& hash, hkUlong id, int tag )
{
	hkAabb dummy; 
	dummy.setEmpty();
	
	return addGeometryHash(source, builder, hash, dummy, 0, hkTransform::getIdentity(), id, tag);
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
