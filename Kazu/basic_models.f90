!-----------------------------------------------------------------------
! module basic models
! 
! This module is a library for calcuration visibilities of basic models
!   1.  elliptical gaussian
!   2.  circular gaussian
!   3.  point source
!
! Writer  : Kazunori Akiyama
! Version : 20120721
! 
! dependency:
!   nothing
!-----------------------------------------------------------------------
module basic_models
    implicit none
    private
    public      :: elliptical_gaussian,&
                   circular_gaussian,&
                   point_source
    real(8) :: pi      = atan(1.0d0)*4.0d0
contains
    !-------------------------------------------------------
    ! Usage
    !    u, v : uvcoverage
    !    p    : parameters
    !        p(1)      : total flux density
    !        p(2),p(3) : central position (r,theta [rad])
    !        p(4)      : FWHM size of major axis
    !        p(5)      : Axial ratio
    !        p(6)      : Position Angle [rad]
    ! output : Vamp, Vphase (rad),Vis
    subroutine elliptical_gaussian(u,v,p,Vcomp,n)
        implicit none
        
        ! arguments
        integer(4), intent(in)    :: n
        real(8),    intent(in)    :: u(1:n),v(1:n)
        real(8),    intent(in)    :: p(1:6)
        complex(8), intent(out) :: Vcomp(1:n)
        
        ! local variables
        real(8)     :: l,m
        real(8)     :: gamma_square(1:n),cosphi,sinphi
        complex(8)  :: i = (0.0d0,1.0d0)
        real(8)     :: Vamp(1:n),Vphase(1:n)
        
        ! Position
        l =   p(2) * sin(p(3))
        m =   p(2) * cos(p(3))
        
        ! sin and cos of position angle
        cosphi = cos(p(6))
        sinphi = sin(p(6))
        
        ! gamma square
        gamma_square = pi**2 * p(4)**2 * ((u*sinphi + v*cosphi)**2 + p(5)**2 * (u*cosphi - v*sinphi) **2)
        Vphase  = - 2 * pi * (u*l + v*m)
        Vcomp   = Vamp * exp(i*Vphase)
    end subroutine elliptical_gaussian
    
    !-------------------------------------------------------
    ! Usage
    !    u, v : uvcoverage
    !    p    : parameters
    !        p(1)      : total flux density
    !        p(2),p(3) : central position (r,theta [rad])
    !        p(4)      : FWHM size of major axis
    ! output : Vamp, Vphase (rad),Vcomp
    subroutine circular_gaussian(u,v,p,Vcomp,n)
        implicit none
        
        ! arguments
        integer(4), intent(in)    :: n
        real(8),    intent(in)    :: u(1:n),v(1:n)
        real(8),    intent(in)    :: p(1:4)
        complex(8), intent(out) :: Vcomp(1:n)
        
        ! local variables
        real(8)     :: l,m
        real(8)     :: gamma_square(1:n)
        complex(8)  :: i = (0.0d0,1.0d0)
        real(8)     :: Vamp(1:n),Vphase(1:n)
        
        ! check
        !write (*,*) "circular gaussian"
        
        ! Position
        l = p(2) * sin(p(3))
        m = p(2) * cos(p(3))
        
        ! gamma square
        gamma_square = pi**2 * p(4)**2 * (u**2 + v**2)
        Vamp    = p(1) * exp(-gamma_square/4d0/log(2d0))
        Vphase  = - 2 * pi * (u*l + v*m)
        Vcomp   = Vamp * exp(i*Vphase)
    end subroutine circular_gaussian
    
    !-------------------------------------------------------
    ! Usage
    !    u, v : uvcoverage
    !    p    : parameters
    !        p(1)      : total flux density
    !        p(2),p(3) : central position (l,m) [rad]
    ! output : Vamp, Vphase (rad),Vcomp
    subroutine point_source(u,v,p,Vcomp,n)
        implicit none
        
        ! arguments
        integer(4), intent(in)    :: n
        real(8),    intent(in)    :: u(1:n),v(1:n)
        real(8),    intent(in)    :: p(1:3)
        complex(8), intent(out) :: Vcomp(1:n)
        
        ! local variables
        real(8)        :: l,m
        complex(8)  :: i = (0.0d0,1.0d0)
        real(8)        :: Vamp(1:n),Vphase(1:n)
        
        ! Position
        l =   p(2) * sin(p(3))
        m =   p(2) * cos(p(3))
        
        ! gamma square
        Vamp   = p(1)
        Vphase =  - 2 * pi * (u*l + v*m)
        Vcomp   = Vamp * exp(i*Vphase)
    end subroutine point_source
    
end module basic_models