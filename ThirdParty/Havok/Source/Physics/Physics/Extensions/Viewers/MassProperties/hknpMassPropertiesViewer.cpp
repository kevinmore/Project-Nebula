/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/Viewers/MassProperties/hknpMassPropertiesViewer.h>

#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>
#include <Common/Base/Math/Matrix/hkMatrix3Util.h>
#include <Common/Base/Types/Color/hkColor.h>

#include <Common/Visualize/hkProcessFactory.h>
#include <Common/Visualize/hkDebugDisplayHandler.h>
#include <Common/Visualize/hkProcessFactory.h>
#include <Common/Visualize/hkDebugDisplay.h>
#include <Common/Visualize/Shape/hkDisplayBox.h>


int hknpMassPropertiesViewer::s_tag = 0;

void HK_CALL hknpMassPropertiesViewer::registerViewer( hkProcessFactory& factory )
{
	s_tag = factory.registerProcess( getName(), create );
}

hkProcess* HK_CALL hknpMassPropertiesViewer::create( const hkArray<hkProcessContext*>& contexts )
{
	return new hknpMassPropertiesViewer( contexts );
}


hknpMassPropertiesViewer::hknpMassPropertiesViewer( const hkArray<hkProcessContext*>& contexts )
:	hknpViewer( contexts ),
	m_bufferSize( 32 * 1024 )
{
}

void hknpMassPropertiesViewer::setBufferSize( int size )
{
	HK_ASSERT2(0x16bc78ba, size >= (int)sizeof(hkDisplayBox), "Buffer size too small" );
	m_bufferSize = size;
}

void hknpMassPropertiesViewer::step( hkReal deltaTime )
{
	if( !m_context )
	{
		return;
	}

	HK_TIMER_BEGIN( "MassPropertiesViewer", this );

	// Allocate buffer for display objects.
	// This avoids large varying size allocations when there are lots of bodies in a world.
	const int bufferCapacity = m_bufferSize / sizeof(hkDisplayBox);
	hkLocalBuffer<hkDisplayBox> displayBoxes( bufferCapacity );
	hkLocalBuffer<hkDisplayGeometry*> displayAabbPtrs( bufferCapacity );
	{
		for( int i=0; i<bufferCapacity; ++i )
		{
			displayAabbPtrs[i] = new (&displayBoxes[i]) hkDisplayBox();
		}
	}

	hkVector4 bias;
	bias.set(1.0f, 1.0001f, 1.0002f, 0.0f);

	const hkReal scale = 1.01f;

	for( int wi=0; wi<m_context->getNumWorlds(); wi++ )
	{
		hknpWorld* world = m_context->getWorld(wi);

		int bufferIndex = 0;
		for( hknpBodyIterator it = world->getBodyIterator(); it.isValid(); it.next() )
		{
			const hknpBody& body = it.getBody();
			if( body.isAddedToWorld() && body.isDynamic() && !body.isKeyframed() )
			{
				const hknpMotion& motion = world->getMotion(body.m_motionId);
				if( motion.m_firstAttachedBodyId != body.m_id )
				{
					continue;	// we only want the first motion
				}

				hkReal invMass = motion.getInverseMass().getReal();

				// Display the mass value
				{
					hkStringBuf str;
					if( invMass != 0.0f )
					{
						str.printf( "%.02f", motion.getMass().getReal() );
					}
					else
					{
						str.printf( "INF" );
					}
					m_displayHandler->display3dText( str.cString(), motion.getCenterOfMassInWorld(), hkColor::MAGENTA, 0, s_tag );
				}

				// Override infinite masses for the calculations below
				invMass = ( invMass > 0.0f ? invMass : 1.0f );

				hkRotation m;
				hkRotation orientation;
				orientation.set( motion.m_orientation );

				hkVector4 invInertiaLocal;
				motion.getInverseInertiaLocal( invInertiaLocal );
				invInertiaLocal.mul( bias );

				hkVector4 halfExtents;
				hkTransform t;
				t.setRotation(body.getTransform().getRotation());
				t.setTranslation(motion.getCenterOfMassInWorld());

				hkVector4Comparison comp = invInertiaLocal.equalZero();
				hkVector4Comparison::Mask mask = comp.getMask<hkVector4ComparisonMask::MASK_XYZ>();
				int count = hkMath::countBitsSet(mask);
				switch (count)
				{
					// No component is zero, so we can compute the "inertia box" from the actual inertia values.
					case 0:
					{
						hkMatrix3 invInertiaLocalDiag;
						hkMatrix3Util::_setDiagonal( invInertiaLocal, invInertiaLocalDiag );
						invInertiaLocalDiag.setMul( orientation, invInertiaLocalDiag );
						m.setMulInverse( invInertiaLocalDiag, orientation );
						m.invertSymmetric();

						hkRotation eigenVec;
						hkVector4 eigenVal;
						m.diagonalizeSymmetricApproximation( eigenVec, eigenVal, 20 );

						hkReal betaSqrd = (eigenVal(0) - eigenVal(1) + eigenVal(2)) * invMass * 6;
						hkReal beta = ((betaSqrd < 0.0f) ? 0.0f : hkMath::sqrt( betaSqrd ));	// Safety check for zeroed elements!
						hkReal gammaSqrd = eigenVal(0) * invMass * 12 - betaSqrd;
						hkReal gamma = ((gammaSqrd < 0.0f) ? 0.0f : hkMath::sqrt( gammaSqrd ));	// Safety check for zeroed elements!
						hkReal alphaSqrd = eigenVal(2) * invMass * 12 - betaSqrd;
						hkReal alpha = ((alphaSqrd < 0.0f) ? 0.0f : hkMath::sqrt( alphaSqrd ));	// Safety check for zeroed elements!

						halfExtents.set( scale * alpha * 0.5f, scale * beta * 0.5f, scale * gamma * 0.5f );
						t.set( eigenVec, motion.getCenterOfMassInWorld() );
						break;
					}

					// 1 component is zero : we will display a plane based on the shape's AABB size;
					case 1:
					{
						// Initialize the box to be the shape's AABB.
						hkAabb aabb;
						body.m_shape->calcAabb(hkTransform::getIdentity(),aabb);
						aabb.getExtents(halfExtents);
						halfExtents.mul(hkSimdReal_Half);
						halfExtents.mul(hkSimdReal::fromFloat(1.05f));

						int axis =
							mask & hkVector4ComparisonMask::MASK_X ?  0 :
							mask & hkVector4ComparisonMask::MASK_Y ?  1 :
													 /* MASK_Z */ 2 ;
						halfExtents.setComponent( axis, hkSimdReal_Eps );
						break;
					}

					// 2 components are zeros, we will show a line along the rotation axis based on the shape's AABB size;
					case 2:
					{
						// Initialize the box to be the shape's AABB.
						hkAabb aabb;
						body.m_shape->calcAabb(hkTransform::getIdentity(),aabb);
						aabb.getExtents(halfExtents);
						halfExtents.mul(hkSimdReal_Half);
						// We need to see this line out of the shape clearly, so scale it a bit more.
						halfExtents.mul(hkSimdReal::fromFloat(1.2f));

						int axis =
							mask == hkVector4ComparisonMask::MASK_XY ?  2 :
							mask == hkVector4ComparisonMask::MASK_XZ ?  1 :
													  /* MASK_YZ */ 0 ;
						hkSimdReal axisLength = halfExtents.getComponent(axis);
						halfExtents.setAll(HK_REAL_EPSILON);
						halfExtents.setComponent( axis, axisLength );
						break;
					}

					// All components are zero, we display nothing.
					case 3:
					{
						continue;
					}

					default:
					{
						HK_ASSERT( 0x87b5ae2f, false );
						continue;
					}
				}

				displayBoxes[bufferIndex].setParameters( halfExtents, t );
				if( ++bufferIndex == bufferCapacity )
				{
					// Flush
					hkArray<hkDisplayGeometry*> displayGeometries( displayAabbPtrs.begin(), bufferIndex, bufferCapacity );
					m_displayHandler->displayGeometry( displayGeometries, hkColor::MAGENTA, 0, s_tag );
					bufferIndex = 0;
				}
			}
		}

		if( bufferIndex > 0 )
		{
			// Flush
			hkArray<hkDisplayGeometry*> displayGeometries( displayAabbPtrs.begin(), bufferIndex, bufferCapacity );
			m_displayHandler->displayGeometry( displayGeometries, hkColor::MAGENTA, 0, s_tag );
		}
	}

	HK_TIMER_END();
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
