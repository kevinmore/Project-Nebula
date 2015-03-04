/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKP_CHAIN_CONSTRAINT_INFO_H
#define HKP_CHAIN_CONSTRAINT_INFO_H


/// Stores values of matrices of LU decomposition of a tri-diagonal constraint matrix.
struct hkpConstraintChainTriple
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_CONSTRAINT_SOLVER, hkpConstraintChainTriple );

	hkReal m_lower;
	hkReal m_diagonal; // XXX todo store inverse inv.
	hkReal m_upper;

	//          [ 0.diagonal                          ]   [ 1       1.upper           ]
	// A = LU = [ 1.lower     1.diagonal              ] * [         1         1.upper ]
	//          [             2.lower      2.diagonal ]   [                   1       ]
	//
	// number in 0.diagonal is the triple index.
	// blanks are zero
};


/// Stores values of matrices of LU decomposition of a block-tri-diagonal constraint matrix.
struct hkpConstraintChainMatrixTriple
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_CONSTRAINT_SOLVER, hkpConstraintChainMatrixTriple );

	hkMatrix3 m_lower;
	hkMatrix3 m_diagonalInv;
	hkMatrix3 m_upper;

	//          [ 0.diagonalInv^-1                              ]   [ 1       1.upper           ]
	// A = LU = [ 1.lower          1.diagonal^-1                ] * [         1         1.upper ]
	//          [                  2.lower        2.diagonal^-1 ]   [                   1       ]
	//
	// number in 0.diagonal is the triple index.
	// blanks are zero
};


class hkpVelocityAccumulator;


class hkpVelocityAccumulatorOffset
{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_CONSTRAINT_SOLVER, hkpVelocityAccumulatorOffset );

		hkpVelocityAccumulatorOffset() {}

		inline hkpVelocityAccumulatorOffset(const hkpVelocityAccumulator* accumulatorBase, const hkpVelocityAccumulator* theOneAccumulator)
		{
			m_offset = static_cast<hkUint32>( hkGetByteOffset(accumulatorBase, theOneAccumulator) );
		}

		inline hkpVelocityAccumulator& getAccumulator(hkpVelocityAccumulator* accumulatorsBase) { return *hkAddByteOffset(accumulatorsBase, m_offset); }
		inline const hkpVelocityAccumulator& getAccumulator(const hkpVelocityAccumulator* accumulatorsBase) { return *hkAddByteOffsetConst(accumulatorsBase, m_offset); }

		hkUint32 m_offset;
};


class hkpConstraintQueryIn;
class hkpConstraintQueryOut;
class hkp1Lin2AngJacobian;


extern "C"
{
	void HK_CALL hkBallSocketChainBuildJacobian( int numConstraints, hkReal tau, hkReal damping, hkReal cfm, hkpVelocityAccumulatorOffset* accumulators, const hkpVelocityAccumulator* accumBase, hkp1Lin2AngJacobian* jacobiansEnd, const hkpConstraintQueryIn &in, hkpConstraintQueryOut &out );

	void HK_CALL hkStabilizedBallSocketChainBuildJacobian( int numConstraints, hkReal tau, hkReal damping, hkReal cfm, hkpVelocityAccumulatorOffset* accumulators, const hkpVelocityAccumulator* accumBase, class hkpChainLinkData* linkDatasEnd, const hkpConstraintQueryIn &in, hkpConstraintQueryOut &out );

	void HK_CALL hkStiffSpringChainBuildJacobian( int numConstraints, hkReal tau, hkReal damping, hkReal cfm, hkpVelocityAccumulatorOffset* accumulators, const hkpVelocityAccumulator* accumBase, hkp1Lin2AngJacobian* jacobiansEnd, const hkpConstraintQueryIn &in, hkpConstraintQueryOut &out );

	void HK_CALL hkBallSocketConstraintBuildJacobian_noSchema_noProj( hkVector4Parameter pivotAWs, hkVector4Parameter pivotBWs, const hkpConstraintQueryIn &in, hkp1Lin2AngJacobian* jac );

	void HK_CALL hkBallSocketConstraintBuildJacobian_noSchema_Proj( hkVector4Parameter pivotAWs, hkVector4Parameter pivotBWs, const hkpConstraintQueryIn &in, hkp1Lin2AngJacobian* jac );

	void HK_CALL hkStabilizedBallSocketConstraintBuildJacobian_noSchema( hkVector4Parameter posA, hkVector4Parameter posB, hkReal maxAllowedErrorDistance, const hkpConstraintQueryIn &in, hkp1Lin2AngJacobian* jac );
}


#endif // HKP_CHAIN_CONSTRAINT_INFO_H

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
