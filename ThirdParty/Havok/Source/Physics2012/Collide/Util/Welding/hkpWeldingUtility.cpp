/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Util/Welding/hkpWeldingUtility.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>

HK_COMPILE_TIME_ASSERT( (sizeof(hkpWeldingUtility::m_sinCosTable)&0xf) == 0);
//
// This method calculates some basic sine and cosine values for each of the 32 hard coded angles.
// Those are stored in an array in the following order:
// cos(accept0), sin(accept0), cos(snap0), cos(snap1), cos(accept1), sin(accept1)
//
void hkpWeldingUtility::initWeldingTable(hkReal edgeNormalSnapDeltaAngle, hkReal triangleNormalSnapDeltaAngle)
{
	// some compile time asserts to validate the layout of the sin-cos-table entries
	HK_COMPILE_TIME_ASSERT( HK_OFFSET_OF(hkpWeldingUtility::SinCosTableEntry,m_cosAccept0) == HK_OFFSET_OF(hkpWeldingUtility::SinCosTableEntry,m_cosAccept1) - 4 * sizeof(hkReal));
	HK_COMPILE_TIME_ASSERT( HK_OFFSET_OF(hkpWeldingUtility::SinCosTableEntry,m_sinAccept0) == HK_OFFSET_OF(hkpWeldingUtility::SinCosTableEntry,m_cosAccept0) + sizeof(hkReal));
	HK_COMPILE_TIME_ASSERT( HK_OFFSET_OF(hkpWeldingUtility::SinCosTableEntry,m_sinAccept1) == HK_OFFSET_OF(hkpWeldingUtility::SinCosTableEntry,m_cosAccept1) + sizeof(hkReal));
	HK_COMPILE_TIME_ASSERT( hkpWeldingUtility::SNAP_0 == 0);
	HK_COMPILE_TIME_ASSERT( hkpWeldingUtility::SNAP_1 == 4);
	HK_COMPILE_TIME_ASSERT( hkpWeldingUtility::WELDING_TYPE_ANTICLOCKWISE == 0);
	HK_COMPILE_TIME_ASSERT( hkpWeldingUtility::WELDING_TYPE_CLOCKWISE == 4);

// #if defined(HK_PLATFORM_SIM) 
// 	// We need to register this static address or otherwise the debug memory will report an unknown address when dma'ing this table to SPU.
// 	if ( hkMemoryRouter::getInstance().isDebugMemory() )
// 	{
// 		hkDebugMemory* debugMemory = static_cast<hkDebugMemory*>( &hkMemoryRouter::getInstance() );
// 		debugMemory->registerStaticAddress(&m_sinCosTable[0], sizeof(m_sinCosTable));
// 	}
// #endif

	for (int i = 0; i < NUM_ANGLES; i++)
	{
		// edgeAngle runs from -180 degree to +180 degree in 360/(NUM_ANGLES-1) degree steps
		hkReal edgeAngle = -HK_REAL_PI + ((2*HK_REAL_PI/hkReal(NUM_ANGLES-1)) * hkReal(i));

		if ( edgeAngle < 0 )
		{
			hkReal flippedEdgeAngle    = HK_REAL_PI - hkMath::fabs(edgeAngle);
			hkReal edgeNormalSnapAngle = flippedEdgeAngle - edgeNormalSnapDeltaAngle;

			//
			// "upper" AcceptSector
			//
			{
				m_sinCosTable[i].m_cosAccept0 = 1.0f;
				m_sinCosTable[i].m_sinAccept0 = 0.0f;
			}

			//
			// TriangleNormal-SnapSector
			//
			if ( edgeNormalSnapAngle > triangleNormalSnapDeltaAngle )
			{
				m_sinCosTable[i].m_cosSnap0 = hkMath::cos(triangleNormalSnapDeltaAngle);
			}
			else
			{
				hkReal tmpAngle = hkMath::max2( edgeNormalSnapAngle, hkReal(0.0f) );
				m_sinCosTable[i].m_cosSnap0 = hkMath::cos(tmpAngle);
			}

			//
			// EdgeNormal-SnapSector
			//
			{
				hkReal tmpAngle = hkMath::max2( edgeNormalSnapAngle, hkReal(0.0f) );
				m_sinCosTable[i].m_cosSnap1 = hkMath::cos(tmpAngle);
			}

			//
			// "lower" AcceptSector
			//
			{
				m_sinCosTable[i].m_cosAccept1 = hkMath::cos(flippedEdgeAngle);
				m_sinCosTable[i].m_sinAccept1 = hkMath::sin(flippedEdgeAngle);
			}

		}
		else
		{
			//
			// "upper" AcceptSector
			//
			{
				m_sinCosTable[i].m_cosAccept0 = hkMath::cos(edgeAngle);
				m_sinCosTable[i].m_sinAccept0 = hkMath::sin(edgeAngle);
			}

			//
			// EdgeNormal-SnapSector
			//
			{
				hkReal tmpAngle = hkMath::min2( (edgeAngle+edgeNormalSnapDeltaAngle), HK_REAL_PI );
				m_sinCosTable[i].m_cosSnap0 = hkMath::cos(tmpAngle);
			}

			//
			// InverseTriangleNormal-SnapSector
			//
			{
				hkReal edgeNormalSnapAngle     = edgeAngle + edgeNormalSnapDeltaAngle;
				hkReal triangleNormalSnapAngle = HK_REAL_PI - triangleNormalSnapDeltaAngle;
				if ( edgeNormalSnapAngle < triangleNormalSnapAngle )
				{
					m_sinCosTable[i].m_cosSnap1 = hkMath::cos(triangleNormalSnapAngle);
				}
				else
				{
					hkReal tmpAngle = hkMath::min2( edgeNormalSnapAngle, HK_REAL_PI );
					m_sinCosTable[i].m_cosSnap1 = hkMath::cos(tmpAngle);
				}
			}

			//
			// "lower" AcceptSector
			//
			{
				m_sinCosTable[i].m_cosAccept1 = -1.0f;
				m_sinCosTable[i].m_sinAccept1 =  0.0f;
			}
		}

	}
}


hkUint16 HK_CALL hkpWeldingUtility::calcScaledWeldingInfo(const hkVector4* vertices, hkUint16 weldingInfo, 
														  hkpWeldingUtility::WeldingType weldingType, hkVector4Parameter scale)
{			
	hkMxSingle<3> normal;
	hkMxVector<3> orthoNormals;
	hkMxSingle<3> scaleMx; scaleMx.setVector(scale);
	hkMxVector<3> scaledEdges;
	{
		// Compute edges
		hkMxVector<3> edges;
		hkMxVector<3> verticesMx;
		verticesMx.moveLoad(vertices);
		edges.setVectorPermutation<hkMxVectorPermutation::SHIFT_LEFT_CYCLIC>(verticesMx);
		edges.setSub(edges, verticesMx);		

		// Scale edges
		scaledEdges.setMul(edges, scaleMx);

		// Calculate triangle normal
		hkVector4 normalSingle;
		normalSingle.setCross(edges.getVector<0>(), edges.getVector<1>());	
		normalSingle.normalize<3>();
		normal.setVector(normalSingle);	

		// Calculate orthonormals
		orthoNormals.setCross(edges, normal);
		orthoNormals.normalize<3,HK_ACC_FULL,HK_SQRT_SET_ZERO>();
	}	

	// Calculate snap vectors	
	hkMxVector<3> snap;
	{
		// Extract welding angle (sin and cos) for each edge
		
		SectorType sectorType = SectorType(weldingType);
		hkMxReal<3> sines;
		hkMxReal<3> cosines;
		{
			const int bitcode = weldingInfo & 0x1F;		
			const hkReal* sinCosTableEntry = &m_sinCosTable[bitcode].m_cosAccept0;
			hkSimdReal sine; sine.setFromFloat(sinCosTableEntry[sectorType]);
			sines.setReal<0>(sine);
			hkSimdReal cosin; cosin.setFromFloat(sinCosTableEntry[sectorType + 1]);
			cosines.setReal<0>(cosin);
		}
		{
			const int bitcode = (weldingInfo >> 5) & 0x1F;
			const hkReal* sinCosTableEntry = &m_sinCosTable[bitcode].m_cosAccept0;
			hkSimdReal sine; sine.setFromFloat(sinCosTableEntry[sectorType]);
			sines.setReal<1>(sine);
			hkSimdReal cosin; cosin.setFromFloat(sinCosTableEntry[sectorType + 1]);
			cosines.setReal<1>(cosin);
		}
		{
			const int bitcode = (weldingInfo >> 10) & 0x1F;
			const hkReal* sinCosTableEntry = &m_sinCosTable[bitcode].m_cosAccept0;
			hkSimdReal sine; sine.setFromFloat(sinCosTableEntry[sectorType]);
			sines.setReal<2>(sine);
			hkSimdReal cosin; cosin.setFromFloat(sinCosTableEntry[sectorType + 1]);
			cosines.setReal<2>(cosin);
		}

		// Calculate snap vectors adding up the normal and orthonormal components
		snap.setZero();
		snap.add(normal);
		snap.mul(sines);
		snap.addMul(cosines, orthoNormals);
	}	

	hkUint16 scaledWeldingInfo = 0;
	{
		// Scale and normalize normal
		hkVector4 scaleInv; scaleInv.setReciprocal(scale);
		hkVector4 scaledNormal; scaledNormal.setMul(scaleInv, normal.getVector());
		scaledNormal.normalize<3>();
		hkMxVector<3> scaledNormals; scaledNormals.setVector<0>(scaledNormal); scaledNormals.setVector<1>(scaledNormal); scaledNormals.setVector<2>(scaledNormal);

		// Scale and normalize orthonormals		
		hkMxVector<3> scaledOrthoNormals; 				
		scaledOrthoNormals.setCross(scaledEdges, scaledNormals);			
		scaledOrthoNormals.normalize<3,HK_ACC_FULL,HK_SQRT_SET_ZERO>();

		// Scale and normalize snap vectors
		hkMxSingle<3> scaleInvMx; scaleInvMx.setVector(scaleInv);
		hkMxVector<3> scaledSnap; scaledSnap.setMul(snap, scaleInvMx);
		scaledSnap.normalize<3,HK_ACC_FULL,HK_SQRT_SET_ZERO>();

		// Calculate welding angles, encode and add them to scaled welding info
		{
			hkMxReal<3> scaledCosines;
			hkMxReal<3> scaledSines;
			scaledSnap.dot<3>(scaledNormals, scaledCosines);
			scaledSnap.dot<3>(scaledOrthoNormals, scaledSines);							

			hkVector4 sines; scaledSines.storePacked(sines);
			hkVector4 cosines; scaledCosines.storePacked(cosines);
			hkVector4 a; hkVector4Util::atan2Approximation(sines,cosines,a);
			HK_ALIGN_REAL(hkReal angle[4]);
			a.store<4>(&angle[0]);
			{
				HK_ASSERT (0xaf730363, (angle[0] >= -HK_REAL_PI) && (angle[0] <= HK_REAL_PI));				
				const int bitcode = calcEdgeAngleBitcode(angle[0]);
				scaledWeldingInfo |= bitcode;
			}
			{
				HK_ASSERT (0xaf730363, (angle[1] >= -HK_REAL_PI) && (angle[1] <= HK_REAL_PI));				
				const int bitcode = calcEdgeAngleBitcode(angle[1]);
				scaledWeldingInfo |= (bitcode << 5);
			}
			{
				HK_ASSERT (0xaf730363, (angle[2] >= -HK_REAL_PI) && (angle[2] <= HK_REAL_PI));				
				const int bitcode = calcEdgeAngleBitcode(angle[2]);
				scaledWeldingInfo |= (bitcode << 10);
			}
		}
	}

	return scaledWeldingInfo;	
}


int HK_CALL hkpWeldingUtility::calcEdgeAngleBitcode(hkReal angle)
{
	int i;
	hkReal refAngle;
	for (i = 0; i < hkpWeldingUtility::NUM_ANGLES; i++)
	{
		// refAngle runs from -180 degree to +180 degree in 360/(NUM_ANGLES-1) degree steps
		refAngle = -HK_REAL_PI + ((2 * HK_REAL_PI / hkReal(hkpWeldingUtility::NUM_ANGLES - 1)) * hkReal(i));
		if (angle <= refAngle)
		{
			if (angle > 0 && angle != refAngle)
			{
				i--;
			}
			break;
		}
	}	

	HK_ASSERT (0xaf730364, (i >= 0) && (i < hkpWeldingUtility::NUM_ANGLES) );
	return i;
}

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
