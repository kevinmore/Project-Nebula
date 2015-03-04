/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


void hknpMxJacobianSorter::MxJacobianElement::flush(hknpConstraintSolverJacobianWriter* jacWriter)
{
#if defined(HK_PLATFORM_SIM) && defined(HK_PLATFORM_SPU)
	if ( m_jacsPpu )
	{
		jacWriter->checkIsStillInSpuBuffer( m_jacsPpu );
	}
#endif
	m_numJacUsed = 0;
}


int hknpMxJacobianSorter::MxJacobianElement::initAndReserveEntry(hkUint32 bodyIdsHashCode, int time,
																 hknpConstraintSolverJacobianWriter* HK_RESTRICT jacWriter,
																 HK_PAD_ON_SPU( hknpMxContactJacobian*)* HK_RESTRICT jacOut,
																 HK_PAD_ON_SPU( hknpMxContactJacobian*)* HK_RESTRICT jacOnPpuOut)
{
	m_usedBodiesHashCode = bodyIdsHashCode;
	m_numJacUsed = 1;
#if defined(HK_JAC_SORTER_LIMIT_ENTRY_AGE)
	m_birthDay = time;
#endif

	hknpMxContactJacobian* jacobians = jacWriter->reserve<hknpMxContactJacobian>();
	m_jacs = jacobians;
	HK_ON_SPU(m_jacsPpu = jacWriter->spuToPpu(jacobians));

	*jacOut = jacobians;
	HK_ON_CPU(*jacOnPpuOut = jacobians);
	HK_ON_SPU(*jacOnPpuOut = m_jacsPpu);

	jacWriter->advance(sizeof(hknpMxContactJacobian));

	//m_jacs->prefetch();
	return 0;
}


int hknpMxJacobianSorter::MxJacobianElement::checkAndReserveEntry(hkUint32 bodyIdsHashCode,
																  HK_PAD_ON_SPU(hknpMxContactJacobian*)* HK_RESTRICT jacOut,
																  HK_PAD_ON_SPU(hknpMxContactJacobian*)* HK_RESTRICT  jacOnPpuOut)
{
	HK_ASSERT( 0xf0cd12c6, m_numJacUsed != 0);

	if( m_usedBodiesHashCode & bodyIdsHashCode )
	{
		return -1;
	}
	int numJac = m_numJacUsed;

	m_usedBodiesHashCode |= bodyIdsHashCode;
	m_numJacUsed = numJac+1;
	*jacOut = m_jacs;
	HK_ON_CPU(*jacOnPpuOut = m_jacs);
	HK_ON_SPU(*jacOnPpuOut = m_jacsPpu);

	return numJac;
}



//
//
//	hknpJacobianConfigulator
//
//

hknpMxJacobianSorter::hknpMxJacobianSorter( hknpConstraintSolverJacobianWriter* HK_RESTRICT jacWriter )
{
	m_jacWriter = jacWriter;
	m_searchOffset = 0;
	m_numOpenElements = 0;
#if defined(HK_JAC_SORTER_LIMIT_ENTRY_AGE)
	m_time = 0;
#endif
}

hknpMxJacobianSorter::~hknpMxJacobianSorter()
{
#ifdef HK_ON_DEBUG
	int numPartiallyFilled = 0;
	for(int i=0; i<m_numOpenElements; i++)
	{
		numPartiallyFilled ++;
#if defined(HK_JAC_SORTER_LIMIT_ENTRY_AGE)
		HK_ON_DEBUG(MxJacobianElement& elem = m_mxJacElement[i]);
		HK_ASSERT( 0xf0456567, elem.m_birthDay >= m_time - MAX_NUM_OPEN_MX_JAC );
#endif
	}
	numPartiallyFilled += 0;
	//HK_REPORT( "NumPartiallyFilled " << numPartiallyFilled);
#endif
}


hkUint32 hknpMxJacobianSorter::calcBodyIdsHashCode( const hknpBody& bodyA, const hknpBody& bodyB )
{
	if ( hknpMxContactJacobian::NUM_MANIFOLDS==1 )	// compile time constant
	{
		return 0;
	}
	else
	{
		int motionIdA = bodyA.m_motionId.value();
		int motionIdB = bodyB.m_motionId.value();

		hkUint32 aFlg = 1 << (motionIdA & 31);
		hkUint32 bFlg = 1 << (motionIdB & 31);
		HK_COMPILE_TIME_ASSERT( hknpBody::IS_STATIC == 1 );
		hkUint32 aIsDynamicMask = ( bodyA.m_flags.anyIsSet( hknpBody::IS_STATIC ) ) + 0xffffffff;
		hkUint32 bIsDynamicMask = ( bodyB.m_flags.anyIsSet( hknpBody::IS_STATIC ) ) + 0xffffffff;

		aFlg &= aIsDynamicMask;
		bFlg &= bIsDynamicMask;

		return aFlg | bFlg;
	}
}

HK_FORCE_INLINE hkUint32 hknpMxJacobianSorter::calcMotionIdsHashCode(hknpMotionId motionIdA, hknpMotionId motionIdB)
{
	if ( hknpMxContactJacobian::NUM_MANIFOLDS==1 )	// compile time constant
	{
		return 0;
	}
	else
	{
		const hkUint32 aFlg = motionIdA.value() ? (1 << (motionIdA.value() & 31)) : 0;
		const hkUint32 bFlg = motionIdB.value() ? (1 << (motionIdB.value() & 31)) : 0;
		return aFlg | bFlg;
	}
}


void hknpMxJacobianSorter::flushElementRange( int startIndex, int numElementsToFlush )
{
	int numElems = m_numOpenElements;

	if (numElementsToFlush)
	{
		MxJacobianElement* s  = &m_mxJacElement[startIndex];
		int k=startIndex;
		for (; k < startIndex+numElementsToFlush; k++)
		{
			s->flush( m_jacWriter );
			s++;
		}

		MxJacobianElement* d		= &m_mxJacElement[startIndex];
		for (; k < numElems; k++)
		{
			*d = *s;
			d++; s++;
		}
		m_numOpenElements = numElems-numElementsToFlush;
	}
}

int hknpMxJacobianSorter::_getJacobianLocation(hkUint32 bodyIdsHashCode, HK_PAD_ON_SPU(hknpMxContactJacobian*)* HK_RESTRICT jacOut, HK_PAD_ON_SPU(hknpMxContactJacobian*)* HK_RESTRICT  jacOnPpuOut)
{

	if ( hknpMxContactJacobian::NUM_MANIFOLDS==1 ) // compile time decision!
	{
		hknpMxContactJacobian* jacobians = m_jacWriter->reserve<hknpMxContactJacobian>();
		*jacOut = jacobians;
		HK_ON_CPU(*jacOnPpuOut = jacobians);
		HK_ON_SPU(*jacOnPpuOut = m_jacWriter->spuToPpu(jacobians));
		m_jacWriter->advance(sizeof(hknpMxContactJacobian));
		return 0;
	}
	else
	{
#if defined(HK_JAC_SORTER_LIMIT_ENTRY_AGE)
		int time = m_time;
#endif
		int searchIndex = m_searchOffset;
		{
			for(int cIdx=searchIndex; cIdx<m_numOpenElements; cIdx++)
			{
				MxJacobianElement* HK_RESTRICT element = &m_mxJacElement[cIdx];
				int jacDstIdx = element->checkAndReserveEntry(bodyIdsHashCode, jacOut, jacOnPpuOut );
				if ( jacDstIdx >=0 )
				{
					m_lastSearchHit = cIdx;
					if ( jacDstIdx == hknpMxContactJacobian::NUM_MANIFOLDS-1  )
					{
						flushElementRange( cIdx, 1);
					}
					return jacDstIdx;
				}
			}
		}

		//
		//	Now we have not found an open mxJac, we have to open a new one
		//  (potentially flushing a half filled one)
		//


		int numOpenElements = m_numOpenElements;
		m_lastSearchHit = 0;

#if defined(HK_JAC_SORTER_LIMIT_ENTRY_AGE)
		//
		//	flush oldest entries
		//
		time++;
		{
			// flush all elements between 0 and numOldElems
			int numOldElems = 0;

			if ( numOpenElements == MAX_NUM_OPEN_MX_JAC)	// buffer full, always flush first
			{
				numOldElems = 1;
			}
			int numElems = m_numOpenElements;
			for (; numOldElems < numElems; numOldElems++)
			{
				const MxJacobianElement* element = &m_mxJacElement[numOldElems];
				if ( element->m_birthDay >= time-MAX_NUM_OPEN_MX_JAC)
				{
					break;
				}
			}
			flushElementRange(0, numOldElems);
			numOpenElements -= numOldElems;
		}
#else
		//	Nothing found, optionally flush oldest entry
		if ( numOpenElements == MAX_NUM_OPEN_MX_JAC)
		{
			// flush oldest entry
			flushElementRange( 0, 1 );
			numOpenElements--;
		}
		const int time = 0;
#endif
		MxJacobianElement* HK_RESTRICT element = &m_mxJacElement[numOpenElements];
		int jacDstIdx = element->initAndReserveEntry( bodyIdsHashCode, time, m_jacWriter, jacOut, jacOnPpuOut);

#if defined(HK_JAC_SORTER_LIMIT_ENTRY_AGE)
		m_time = time;
#endif
		m_numOpenElements = numOpenElements+1;
		return jacDstIdx;
	}
}

void hknpMxJacobianSorter::hintUsingSameBodies()
{
	m_searchOffset = m_lastSearchHit+1;
}

void hknpMxJacobianSorter::resetHintUsingSameBodies()
{
	m_searchOffset = 0;
}

void hknpMxJacobianSorter::flush()
{
	flushElementRange( 0, m_numOpenElements);
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
