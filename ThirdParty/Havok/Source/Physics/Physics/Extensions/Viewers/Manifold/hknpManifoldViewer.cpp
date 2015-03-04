/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/Viewers/Manifold/hknpManifoldViewer.h>

#include <Common/Base/Math/Vector/hkVector4Util.h>
#include <Common/Base/Types/Color/hkColor.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/Internal/GeometryProcessing/ConvexHull/hkgpConvexHull.h>
#include <Common/Visualize/hkProcessFactory.h>

#include <Physics/Physics/Collide/hknpCdBody.h>
#include <Physics/Physics/Collide/NarrowPhase/hknpManifold.h>
#include <Physics/Physics/Dynamics/World/Events/hknpEventDispatcher.h>
#include <Physics/Physics/Collide/NarrowPhase/Cache/hknpCollisionCache.h>


int hknpManifoldViewer::s_tag = 0;

void HK_CALL hknpManifoldViewer::registerViewer( hkProcessFactory& factory )
{
	s_tag = factory.registerProcess( getName(), create );
}

hkProcess* HK_CALL hknpManifoldViewer::create( const hkArray<hkProcessContext*>& contexts )
{
	return new hknpManifoldViewer( contexts );
}


hknpManifoldViewer::hknpManifoldViewer( const hkArray<hkProcessContext*>& contexts )
:	hknpViewer( contexts )
{
}

hknpManifoldViewer::~hknpManifoldViewer()
{
	if( m_context )
	{
		for( int i=0; i < m_context->getNumWorlds(); ++i )
		{
			worldRemovedCallback( m_context->getWorld(i) );
		}
	}
}

void hknpManifoldViewer::worldAddedCallback( hknpWorld* world )
{
	world->m_modifierManager->incrementGlobalBodyFlags( hknpBody::RAISE_MANIFOLD_PROCESSED_EVENTS );
	world->getEventSignal( hknpEventType::MANIFOLD_PROCESSED ).subscribe( this, &hknpManifoldViewer::onManifoldProcessedEvent, "ManifoldViewer");
}

void hknpManifoldViewer::worldRemovedCallback( hknpWorld* world )
{
	world->m_modifierManager->decrementGlobalBodyFlags( hknpBody::RAISE_MANIFOLD_PROCESSED_EVENTS );
	world->getEventSignal( hknpEventType::MANIFOLD_PROCESSED ).unsubscribeAll( this );
}

void hknpManifoldViewer::onManifoldProcessedEvent( const hknpEventHandlerInput& input, const hknpEvent& event )
{
	HK_TIME_CODE_BLOCK( "ManifoldViewer", this );

	const hknpManifoldProcessedEvent& mpEvent = event.asManifoldProcessedEvent();

	// Choose a color (green for normal case, red if no Jacobian)
	const int color = mpEvent.m_manifoldCache->m_manifoldSolverInfo.m_contactJacobian ?
		0x6666ff66 : 0x66ff6666;

	const hkcdManifold4& manifold = mpEvent.m_manifold;
	int numContactPoints = mpEvent.m_numContactPoints;
	HK_ASSERT( 0x7e0d6f72, numContactPoints <= 4 );

	// Convexify the vertices if necessary
	int remap[4] = { 0, 1, 2, 3 };
	if( numContactPoints == 4 )
	{
		// Get source vertices
		const hkVector4* HK_RESTRICT vSrc = mpEvent.m_manifold.m_positions;

		// Compute their AABB
		hkAabb aabb;
		aabb.setEmpty();
		aabb.includePoint(vSrc[0]);
		aabb.includePoint(vSrc[1]);
		aabb.includePoint(vSrc[2]);
		aabb.includePoint(vSrc[3]);

		// Remap such that the first two vertices are furthest apart
		{
			hkVector4 vExtents; aabb.getExtents(vExtents);
			hkVector4 vMaxExtent; vMaxExtent.setAll(vExtents.horizontalMax<3>());

			hkVector4Comparison cmp; cmp.set<hkVector4ComparisonMask::MASK_XYZ>();
			cmp.setAnd(cmp, vExtents.greaterEqual(vMaxExtent));

			const int maxDispersionAxis = cmp.getIndexOfFirstComponentSet();

			// Transpose points to get the maximum dispersed coordinates
			hkVector4 vCoords;
			{
				hkFourTransposedPoints pts;
				pts.set(vSrc[0], vSrc[1], vSrc[2], vSrc[3]);
				vCoords = pts.m_vertices[maxDispersionAxis];
			}

			// Get indices of the min and max points on the maximum dispersion axis
			hkVector4 coordMin; coordMin.setAll(vCoords.horizontalMin<4>());
			hkVector4 coordMax; coordMax.setAll(vCoords.horizontalMax<4>());
			const int idxA = vCoords.lessEqual(coordMin).getIndexOfFirstComponentSet();
			const int idxB = vCoords.greaterEqual(coordMax).getIndexOfLastComponentSet();

			// Compute the other two indices
			const int idxCD = 0xF & (~((1 << idxA) | (1 << idxB)));
			const int idxC = 31 - hkMath::countLeadingZeros(idxCD);
			const int idxD = hkMath::countTrailingZeros(idxCD);

			remap[0] = idxA;
			remap[1] = idxB;
			remap[2] = idxC;
			remap[3] = idxD;
		}

		const hkVector4 vA = vSrc[remap[0]];
		const hkVector4 vB = vSrc[remap[1]];
		hkVector4 vAB; vAB.setSub(vB, vA);

		// Compute third vertex as furthest apart from the first 2
		{
			// Compute a perpendicular vector
			hkVector4 vPerp; hkVector4Util::calculatePerpendicularVector(vAB, vPerp);

			// Pick the furthest point along this vector
			hkVector4 vC; vC.setSub(vSrc[remap[2]], vA); vC.setInt24W(remap[2]);
			hkVector4 vD; vD.setSub(vSrc[remap[3]], vA); vD.setInt24W(remap[3]);

			hkSimdReal dotC; dotC.setAbs(vC.dot<3>(vPerp));
			hkSimdReal dotD; dotD.setAbs(vD.dot<3>(vPerp));

			const hkVector4Comparison cmp = dotC.greater(dotD);

			hkVector4 newC; newC.setSelect(cmp, vC, vD);
			hkVector4 newD; newD.setSelect(cmp, vD, vC);

			remap[2] = newC.getInt24W();
			remap[3] = newD.getInt24W();
		}

		// Last vertex
		{
			const hkVector4 vC = vSrc[remap[2]];
			const hkVector4 vD = vSrc[remap[3]];

			hkVector4 vBC, vCA; vBC.setSub(vC, vB); vCA.setSub(vA, vC);
			hkVector4 vAD, vBD, vCD; vAD.setSub(vD, vA); vBD.setSub(vD, vB); vCD.setSub(vD, vC);
			hkVector4 vN; vN.setCross(vCA, vAB);
			hkVector4 vABD, vBCD, vCAD; hkVector4Util::cross_3vs1(vAD, vBD, vCD, vN, vABD, vBCD, vCAD);
			hkVector4 vDots; hkVector4Util::dot3_3vs3(vAB, vABD, vBC, vBCD, vCA, vCAD, vDots);
			const int cmpMask = (vDots.lessZero().getMask() & 7) << 1;

			const int idxA = remap[(0x0C00 >> cmpMask) & 3];
			const int idxB = remap[(0x55DD >> cmpMask) & 3];
			const int idxC = remap[(0xBAB6 >> cmpMask) & 3];
			const int idxD = remap[(0xBBAA >> cmpMask) & 3];

			remap[0] = idxA;
			remap[1] = idxB;
			remap[2] = idxC;
			remap[3] = idxD;
		}

		//HK_ASSERT(0x70cdb373, ((1<<remap[0]) | (1<<remap[1]) | (1<<remap[2]) | (1<<remap[3])) == 0xf );

		// Remove duplicates, for easier displaying
		if( remap[2] == remap[3] )
		{
			numContactPoints--;
		}
	}

	// Calculate contact point positions on each body
	hkVector4 pointsA[4];
	hkVector4 pointsB[4];
	{
		for( int i = 0; i < numContactPoints; i++ )
		{
			const int j = remap[i];
			pointsB[i] = manifold.m_positions[j];
			hkVector4 offset; offset.setMul( manifold.getDistance(j), manifold.m_normal );
			pointsA[i].setAdd( manifold.m_positions[j], offset );
		}
	}

	// Draw connection
	{
		hkVector4 centroidA; centroidA.setZero();
		hkVector4 centroidB; centroidB.setZero();
		for( int i = 0; i < numContactPoints; ++i )
		{
			centroidA.add( pointsA[i] );
			centroidB.add( pointsB[i] );
		}
		hkSimdReal r; r.setReciprocal( hkSimdReal::fromInt32(numContactPoints) );
		centroidA.mul( r );
		centroidB.mul( r );

		m_displayHandler->displayLine( centroidA, centroidB, color, 0, s_tag );
	}

	// Draw manifold borders
	{
		switch( numContactPoints )
		{
		case 1:
			{
				m_displayHandler->displayPoint( pointsA[0], color, 0, s_tag );
				m_displayHandler->displayPoint( pointsB[0], color, 0, s_tag );
			}
			break;

		case 2:
			{
				m_displayHandler->displayLine( pointsA[0], pointsA[1], color, 0, s_tag );
				m_displayHandler->displayLine( pointsB[0], pointsB[1], color, 0, s_tag );
			}
			break;

		case 3:
		case 4:
			{
				for( int i=numContactPoints-1, j=0; j<numContactPoints; i=j++ )
				{
					m_displayHandler->displayLine( pointsA[i], pointsA[j], color, 0, s_tag );
					m_displayHandler->displayLine( pointsB[i], pointsB[j], color, 0, s_tag );
				}
			}
			break;

		default:
			break;
		}
	}

	// Draw contact points
	
	if(0)
	{
		for( int i=0; i<numContactPoints; i++ )
		{
			m_displayHandler->displayPoint( pointsA[i], color, 0, s_tag );
			m_displayHandler->displayPoint( pointsB[i], color, 0, s_tag );
		}
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
