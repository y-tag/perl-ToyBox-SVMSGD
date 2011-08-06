package ToyBox::SVMSGDQN;

use strict;
use warnings;

use Data::Dumper;

our $VERSION = '0.0.1';

sub new {
    my $class = shift;
    my $self = {
        dnum => 0,
        data => [],
        lindex => {},
        rlindex => {},
        weight => {},
    } ;
    bless $self, $class;
}

sub add_instance {
    my ($self, %params) = @_;

    my $attributes = $params{attributes} or die "No params: attributes";
    my $label      = $params{label}     or die "No params: label";
    die "attributes is not hash ref"   unless ref($attributes) eq 'HASH';
    die "attributes is empty hash ref" unless keys %$attributes;
    $label = [$label] unless ref($label) eq 'ARRAY';

    my %copy_attr = %$attributes;

    foreach my $l (@$label) {
        if (! defined($self->{lindex}{$l})) {
            my $index = scalar(keys %{$self->{lindex}});
            $self->{lindex}{$l} = $index;
            $self->{rlindex}{$index} = $l;
        }
        my $datum = {f => \%copy_attr, l => $self->{lindex}{$l}};
        push(@{$self->{data}}, $datum);
        $self->{dnum}++;
    }
}

sub train {
    my ($self, %params) = @_;

    my $lambda = $params{lambda} || 1e-1;
    my $t0     = $params{t0}     || 1;
    my $T      = $params{T}      || $self->{dnum};
    my $skip   = $params{skip}   || 16;
    die "lambda is le 0" unless scalar($lambda) > 0;
    die "t0 is le 0" unless scalar($t0) > 0;
    die "T is le 0" unless scalar($T) > 0;
    die "skip is le 0" unless scalar($skip) > 0;

    my $progress_cb = $params{progress_cb};
    my $verbose = 0;
    $verbose = 1 if defined($progress_cb) && $progress_cb eq 'verbose';

    my $weight = $self->{weight};
    my $dnum   = $self->{dnum};
    my $B = {};
    my $updateB = 0;
    my $r = 2;

    print STDERR "learn start: " if $verbose;
    my $t = 0;
    my $count = $skip;
    while ($t < $T) {
        my $datum = $self->{data}[int(rand($dnum-1))];
        my $attributes = $datum->{f};
        my $label = $datum->{l};

        my $scores = {};
        while (my ($f, $val) = each %$attributes) {
            next unless $weight->{$f};
            while (my ($l, $w) = each %{$weight->{$f}}) {
                $scores->{$l} += $val * $w;
            }
        }
        foreach my $l (keys %{$self->{rlindex}}) {
            my $y = ($label eq $l) ? 1 : -1;
            my $score = $scores->{$l} || 0;
            my $dloss = 1 - $y * $score > 0 ? 1 : 0;
            next unless $dloss > 0;

            my $coe = $y / ($t0 + $t);
            while (my ($f, $v) = each %$attributes) {
                $B->{$f}{$l} = (1 / $lambda) unless defined($B->{$f}{$l});
                $weight->{$f}{$l} += $coe * $B->{$f}{$l} * $v;
            }
        }

        if ($updateB == 1) {
            my $scores2 = {};
            while (my ($f, $val) = each %$attributes) {
                next unless $weight->{$f};
                while (my ($l, $w) = each %{$weight->{$f}}) {
                    $scores2->{$l} += $val * $w;
                }
            }

            my $lower_bound = 1 / (1e2 * $lambda);
            foreach my $f (keys %$B) {
                foreach my $l (keys %{$B->{$f}}) {
                    my $oldB = $B->{$f}{$l} || (1 / $lambda);
                    $B->{$f}{$l} *= 1 - (2 / $r) if defined($B->{$f}{$l});
                    my $y = ($label eq $l) ? 1 : -1;
                    my $score1 = $scores->{$l} || 0;
                    my $dloss1 = 1 - $y * $score1 > 0 ? 1 : 0;
                    my $score2 = $scores2->{$l} || 0;
                    my $dloss2 = 1 - $y * $score2 > 0 ? 1 : 0;
                    my $diff_loss = $dloss2 - $dloss1;

                    if ($diff_loss != 0) {
                        my $denom = $lambda + (($t + $t0) * $diff_loss) / (-1 * $dloss1 * $oldB);
                        $B->{$f}{$l} += 2 / ($r * $denom);
                    } else {
                        $B->{$f}{$l} += 2 / ($r * $lambda);
                    }
                    $B->{$f}{$l} = $lower_bound if $lower_bound > $B->{$f}{$l};
                }
            }
            $r += 1;
            $updateB = 0;
        }
        $count -= 1;
        if ($count <= 0) {
            my $coe = $skip * $lambda / ($t0 + $t);
            foreach my $f (keys %$weight) {
                foreach my $l (keys %{$weight->{$f}}) {
                    $weight->{$f}{$l} *= (1 - $coe * $B->{$f}{$l});
                }
            }
            $count = $skip;
            $updateB = 1;
        }

        if ($verbose) {
            if    ($t % 1000 == 0) { print STDERR $t; }
            elsif ($t % 100  == 0) { print STDERR "."; }
        }
        $t += 1;
    }
    $self->{weight} = $weight;
    print STDERR "done($t)\n" if $verbose;

    1;
}

sub predict {
    my ($self, %params) = @_;

    my $attributes = $params{attributes} or die "No params: attributes";
    die "attributes is not hash ref"   unless ref($attributes) eq 'HASH';
    die "attributes is empty hash ref" unless keys %$attributes;

    my $weight = $self->{weight};
    my $rlindex = $self->{rlindex};

    my $scores = {};
    while (my ($f, $val) = each %$attributes) {
        while (my ($l, $w) = each %{$weight->{$f}}) {
            $scores->{$rlindex->{$l}} += $val * $w;
        }
    }

    $scores;
}

sub labels {
    my $self = shift;
    keys %{$self->{lindex}};
}

sub data_num {
    my $self = shift;
    $self->{dnum};
}

1;
__END__


=head1 NAME

ToyBox::SVMSGDQN - Classifier using SVMSGDQN Algorithm

=head1 SYNOPSIS

  use ToyBox::SVMSGDQN;

  my $svm= ToyBox::SVMSGDQN->new();
  
  $svm->add_instance(
      attributes => {a => 2, b => 3},
      label => 'positive'
  );
  
  $svm->add_instance(
      attributes => {c => 3, d => 1},
      label => 'negative'
  );
  
  $svm->train(T => $svm->data_num(),
              lambda => 1.0,
              t0 => 10,
              skip => 16,
              progress_cb => 'verbose');
  
  my $score = $svm->predict(
                  attributes => {a => 1, b => 1, d => 1, e =>1}
              );

=head1 DESCRIPTION

=head1 AUTHOR

TAGAMI Yukihiro <tagami.yukihiro@gmail.com>

=head1 LICENSE

This library is distributed under the term of the MIT license.

L<http://opensource.org/licenses/mit-license.php>

