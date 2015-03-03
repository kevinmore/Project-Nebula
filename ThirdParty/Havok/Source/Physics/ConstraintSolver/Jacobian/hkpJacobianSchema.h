/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKP_JACOBIAN_SCHEMA_H
#define HKP_JACOBIAN_SCHEMA_H

#include <Physics/ConstraintSolver/Solve/hkpSolve.h>
#include <Physics/ConstraintSolver/Jacobian/hkpJacobianElement.h>


// With normal compression you can still take jacobians as they are, and use them for jacobian calculatsions.
// They have other data stored in their least significant bits.


#	define HK_SCHEMA_INIT(schema, jacType, jacName)\
	/*register*/ jacType* HK_RESTRICT jacName = reinterpret_cast<jacType*>(schema);  // todo: use a register

///////////////////////////////////////////////////////////////////////////
//
//  Decompression macros
//
//////////////////////////////////////////////////////////////////////////

#	define HK_SCHEMA_UNPACK_MOTOR(schema, jacType, motorSolverInfoType, jac, motorInfo) const motorSolverInfoType& motorInfo = schema.m_motorSolverInfo;

#	define HK_SCHEMA_GET_MEMBER(schema, memberName) (schema).m_##memberName

#	define HK_SCHEMA_UNPACK_JACOBIAN(schema, jacType, jacDest, jacExtPtr)\
		jacExtPtr = &(schema)->m_jac;
#	define HK_SCHEMA_UNPACK_JACOBIAN_0(schema, jacType, jacDest, jacExtPtr)\
		jacExtPtr = &(schema)->m_jac0;
#	define HK_SCHEMA_UNPACK_JACOBIAN_1(schema, jacType, jacDest, jacExtPtr)\
		jacExtPtr = &(schema)->m_jac1;
#	define HK_SCHEMA_UNPACK_JACOBIAN_ULTRA(schema, jacType, jacDest, jacExtPtr)\
		   HK_SCHEMA_UNPACK_JACOBIAN(schema, jacType, jacDest, jacExtPtr)
#	define HK_SCHEMA_UNPACK_JACOBIAN_AND_SR(schema, jacType, jacDest, jacExtPtr, dsr)\
		jacExtPtr = &(schema)->m_jac;\
		(schema)->unpackSolverResultsPtr(dsr);
#	define HK_SCHEMA_UNPACK_2_JACOBIANS_AND_SR(schema, jacType, jacDest, jacExtPtr, dsr)\
		jacExtPtr = &(schema)->m_jac0;\
		(schema)->unpackSolverResultsPtr(dsr);



////////////////////////////////////////////////////////////////////////////
//
//
//
//////////////////////////////////////////////////////////////////////////

#define HK_SET_SCHEMA_TYPE(schema, type) hkpJacobianSchema::setType(reinterpret_cast<hkpJacobianSchema*>(schema), hkpJacobianSchema::type)
#define HK_SET_SCHEMA_TYPE_ACCURATE(schema, type) hkpJacobianSchema::setTypeAccurate(reinterpret_cast<hkpJacobianSchema*>(schema), hkpJacobianSchema::type)
#define HK_GET_SCHEMA_TYPE(schema)       hkpJacobianSchema::getType(reinterpret_cast<const hkpJacobianSchema*>(schema))
#define HK_AS_JACOBIAN_SCHEMA(schema) reinterpret_cast<hkpJacobianSchema*>(schema)
#define HK_AS_JACOBIAN_SCHEMA_CONST(schema) reinterpret_cast<const hkpJacobianSchema*>(schema)

	/// the base class of all constraint commands passed to the solver
class hkpJacobianSchema
{
	public:
		enum SchemaType
		{
			// Note: we assume 	SCHEMA_TYPE_END will be 0 in hkdynamics code (where we do not want to include this file).
			// Note: the order of the schemas CANNOT be altered.
			// Do not change this!
			SCHEMA_TYPE_END = 0,
			SCHEMA_TYPE_HEADER,

				// control schemas
			SCHEMA_TYPE_GOTO, // = 2
			SCHEMA_TYPE_SHIFT_SOLVER_RESULTS_POINTER,

				// start of simple 'fixed-size schemas
			SCHEMA_TYPE_SIMPLE_BEGIN,

				// linear schemas
			SCHEMA_TYPE_1D_BILATERAL, // = 5
			SCHEMA_TYPE_1D_BILATERAL_WITH_IMPULSE_LIMIT, // used by the ball-socked with impulse limit
			SCHEMA_TYPE_1D_BILATERAL_USER_TAU,
			SCHEMA_TYPE_1D_LINEAR_LIMITS,
			SCHEMA_TYPE_1D_FRICTION,
			SCHEMA_TYPE_1D_LINEAR_MOTOR,

				// linear pulley schema
			SCHEMA_TYPE_1D_PULLEY,

				// angular schemas
			SCHEMA_TYPE_1D_ANGULAR, // = 12
			SCHEMA_TYPE_1D_ANGULAR_LIMITS,
			SCHEMA_TYPE_1D_ANGULAR_FRICTION,
			SCHEMA_TYPE_1D_ANGULAR_MOTOR,

				// Stable (in-place solver) schemas
			SCHEMA_TYPE_STABLE_BALLSOCKET,
			SCHEMA_TYPE_NP_STABLE_BALLSOCKET,
			SCHEMA_TYPE_STABLE_STIFFSPRING,
			SCHEMA_TYPE_NP_STABLE_STIFFSPRING,

				// contact schemas
			SCHEMA_TYPE_SINGLE_CONTACT, // = 20
			SCHEMA_TYPE_SINGLE_CONTACT_WITH_ACCESSOR_CHECK,

			SCHEMA_TYPE_SINGLE_LIMIT_CONTACT, // = 22
			SCHEMA_TYPE_SINGLE_LIMIT_CONTACT_WITH_ACCESSOR_CHECK,

			SCHEMA_TYPE_PAIR_CONTACT,
			SCHEMA_TYPE_PAIR_CONTACT_WITH_ACCESSOR_CHECK,
			SCHEMA_TYPE_2D_FRICTION,
			SCHEMA_TYPE_3D_FRICTION,
			SCHEMA_TYPE_2D_ROLLING_FRICTION,

				// modifier schemas
			SCHEMA_TYPE_SET_MASS,	// = 29
			SCHEMA_TYPE_ADD_VELOCITY,
			SCHEMA_TYPE_SET_CENTER_OF_MASS,

			SCHEMA_TYPE_3D_ANGULAR,
			SCHEMA_TYPE_NP_3D_ANGULAR,
			SCHEMA_TYPE_DEFORMABLE_LIN_3D,
			SCHEMA_TYPE_DEFORMABLE_ANG_3D,
			SCHEMA_TYPE_NP_DEFORMABLE_LIN_3D,
			SCHEMA_TYPE_NP_DEFORMABLE_ANG_3D,

				// end of simple 'fixed-size' schemas; synchronize with hkSolverExport::exportImpulsesAndRhs
			SCHEMA_TYPE_SIMPLE_END,

				// chain schemas
			SCHEMA_TYPE_STIFF_SPRING_CHAIN,	// = 39
			SCHEMA_TYPE_BALL_SOCKET_CHAIN,
			SCHEMA_TYPE_STABILIZED_BALL_SOCKET_CHAIN,
			SCHEMA_TYPE_POWERED_CHAIN,

			SCHEMA_TYPE_WHEEL_FRICTION,

			SCHEMA_TYPE_MAX // = 44
		};

			// 8-bit enumeration type
		typedef hkEnum<SchemaType, hkInt8> JointSchemaType;



#	if HK_ENDIAN_LITTLE
#		define HK_SCHEMA_TYPE_OFFSET 0
#	elif HK_ENDIAN_BIG
#		define HK_SCHEMA_TYPE_OFFSET 3
#	else
#		error unknown endianness
#	endif


		public:

			static HK_FORCE_INLINE SchemaType	getType(const hkpJacobianSchema* schema){ return SchemaType( reinterpret_cast<const hkUint8*>(schema)[HK_SCHEMA_TYPE_OFFSET] ) ; }
			static HK_FORCE_INLINE int			getSize(const hkpJacobianSchema* schema)
			{
				const SchemaType type = hkpJacobianSchema::getType(schema);
				const int size = s_schemaSize[int(type)];
				HK_ASSERT2(0xad67ab9a, size != 0 && size != 0xff, "Size not defined for the specified hkpJacobianSchema type.");
				HK_ASSERT2(0xad67ab9b, 0 == (size & (HK_REAL_ALIGNMENT-1)) , "Size is not a multiple of SIMD alignment.");
				return size;
			}
			static HK_FORCE_INLINE int getNumSolverElemTemps(const hkpJacobianSchema* schema)
			{
				const SchemaType type = hkpJacobianSchema::getType(schema);
				const int numSolverElemTemps = s_schemaNumSolverElemTemps[int(type)];
				HK_ASSERT2(0xad67ab9a, numSolverElemTemps >= 0 && numSolverElemTemps <= 3+1, "Internal error: sizes for chain schemas not defined yet..."); // max is 3d friction: 3sr + 1friction elem temp
				return numSolverElemTemps;
			}

		public:

			static HK_FORCE_INLINE void			setType(hkpJacobianSchema* schema, SchemaType type)
			{
				// Reduce likelihood of denormals numbers on pc.
#			if defined (HK_PLATFORM_WIN32) || defined (HK_PLATFORM_XBOX)
				*reinterpret_cast<hkReal*>(schema) += HK_REAL_EPSILON;
#			endif
				*reinterpret_cast<JointSchemaType*>( hkAddByteOffset(schema, HK_SCHEMA_TYPE_OFFSET) ) = type;
			}

			// Schemas that use this version of setType must use clearTypeFromReal() later to avoid denormals
			static HK_FORCE_INLINE void			setTypeAccurate(hkpJacobianSchema* schema, SchemaType type)
			{
				*reinterpret_cast<JointSchemaType*>( hkAddByteOffset(schema, HK_SCHEMA_TYPE_OFFSET) ) = type;
			}

			static HK_FORCE_INLINE void			clearTypeFromReal(hkReal* real)
			{
				*reinterpret_cast<hkUint8*>( hkAddByteOffset(real, HK_SCHEMA_TYPE_OFFSET) ) = 0;
			}

#if defined(HK_REAL_IS_DOUBLE)
			static hkUint16 s_schemaSize[];
#else
			static hkUint8 s_schemaSize[];
#endif
			static hkUint8 s_schemaNumSolverElemTemps[];

#		if !defined(HK_PLATFORM_SPU) || defined(HK_PLATFORM_WIN32)
			static void HK_CALL verifySchemaInfoArrays();
#		endif

};


#endif // HKP_JACOBIAN_SCHEMA_H

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
